# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import inspect
import contextlib
import json
import logging
import os
import re
import tempfile
import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypedDict, Union, cast

from openai import OpenAI, AzureOpenAI
from azure.ai.evaluation._legacy._adapters._constants import LINE_NUMBER
from azure.ai.evaluation._legacy._adapters.entities import Run
import pandas as pd

from azure.ai.evaluation._common.math import list_mean_nan_safe, apply_transform_nan_safe
from azure.ai.evaluation._common.utils import validate_azure_ai_project, is_onedp_project
from azure.ai.evaluation._exceptions import ErrorBlame, ErrorCategory, ErrorTarget, EvaluationException

from azure.ai.evaluation._aoai.aoai_grader import AzureOpenAIGrader

from .._constants import (
    CONTENT_SAFETY_DEFECT_RATE_THRESHOLD_DEFAULT,
    EVALUATION_PASS_FAIL_MAPPING,
    EvaluationMetrics,
    DefaultOpenEncoding,
    Prefixes,
    _InternalEvaluationMetrics,
    BINARY_AGGREGATE_SUFFIX,
    DEFAULT_OAI_EVAL_RUN_NAME,
)
from .._model_configurations import AzureAIProject, EvaluationResult, EvaluatorConfig
from .._user_agent import UserAgentSingleton
from ._batch_run import (
    EvalRunContext,
    CodeClient,
    ProxyClient,
    TargetRunContext,
    RunSubmitterClient,
)
from ._utils import (
    _apply_column_mapping,
    _log_metrics_and_instance_results,
    _trace_destination_from_project_scope,
    _write_output,
    DataLoaderFactory,
    _log_metrics_and_instance_results_onedp,
)
from ._batch_run.batch_clients import BatchClient, BatchClientRun

from ._evaluate_aoai import (
    _begin_aoai_evaluation,
    _split_evaluators_and_grader_configs,
    _get_evaluation_run_results,
    OAIEvalRunCreationInfo,
)

LOGGER = logging.getLogger(__name__)

# For metrics (aggregates) whose metric names intentionally differ from their
# originating column name, usually because the aggregation of the original value
# means something sufficiently different.
# Note that content safety metrics are handled separately.
METRIC_COLUMN_NAME_REPLACEMENTS = {
    "groundedness_pro_label": "groundedness_pro_passing_rate",
}


class __EvaluatorInfo(TypedDict):
    result: pd.DataFrame
    metrics: Dict[str, Any]
    run_summary: Dict[str, Any]


class __ValidatedData(TypedDict):
    """
    Simple dictionary that contains ALL pre-processed data and
    the resultant objects that are needed for downstream evaluation.
    """

    evaluators: Dict[str, Callable]
    graders: Dict[str, AzureOpenAIGrader]
    input_data_df: pd.DataFrame
    column_mapping: Dict[str, Dict[str, str]]
    target_run: Optional[BatchClientRun]
    batch_run_client: BatchClient
    batch_run_data: Union[str, os.PathLike, pd.DataFrame]


def _aggregate_other_metrics(df: pd.DataFrame) -> Tuple[List[str], Dict[str, float]]:
    """Identify and average various metrics that need to have the metric name be replaced,
    instead of having the metric match the originating column name.
    :param df: The dataframe of evaluation results.
    :type df: ~pandas.DataFrame
    :return: A tuple; the first element is a list of dataframe columns that were aggregated,
        and the second element is a dictionary of resultant new metric column names and their values.
    :rtype: Tuple[List[str], Dict[str, float]]
    """
    renamed_cols = []
    metric_columns = {}
    for col in df.columns:
        metric_prefix = col.split(".")[0]
        metric_name = col.split(".")[1]
        if metric_name in METRIC_COLUMN_NAME_REPLACEMENTS:
            renamed_cols.append(col)
            new_col_name = metric_prefix + "." + METRIC_COLUMN_NAME_REPLACEMENTS[metric_name]
            col_with_numeric_values = cast(List[float], pd.to_numeric(df[col], errors="coerce"))
            try:
                metric_columns[new_col_name] = round(list_mean_nan_safe(col_with_numeric_values), 2)
            except EvaluationException:  # only exception that can be cause is all NaN values
                msg = f"All score evaluations are NaN/None for column {col}. No aggregation can be performed."
                LOGGER.warning(msg)

    return renamed_cols, metric_columns


# pylint: disable=line-too-long
def _aggregate_content_safety_metrics(
    df: pd.DataFrame, evaluators: Dict[str, Callable]
) -> Tuple[List[str], Dict[str, float]]:
    """Find and aggregate defect rates for content safety metrics. Returns both a list
    of columns that were used to calculate defect rates and the defect rates themselves.

    :param df: The dataframe of evaluation results.
    :type df: ~pandas.DataFrame
    :param evaluators:  A dictionary mapping of strings to evaluator classes. This is used to identify
        content safety metrics, since they should start with a string that matches an evaluator name.
    :type evaluators: Dict[str, type]
    :return: A tuple; the first element is a list of dataframe columns that were used to calculate defect rates,
        and the second element is a dictionary of defect column names and defect rates.
    :rtype: Tuple[List[str], Dict[str, float]]
    """
    content_safety_metrics = [
        EvaluationMetrics.SEXUAL,
        EvaluationMetrics.SELF_HARM,
        EvaluationMetrics.HATE_UNFAIRNESS,
        EvaluationMetrics.VIOLENCE,
    ]
    content_safety_cols = []
    for col in df.columns:
        evaluator_name = col.split(".")[0]
        metric_name = col.split(".")[1]
        if evaluator_name in evaluators:
            # Check the namespace of the evaluator
            module = inspect.getmodule(evaluators[evaluator_name])
            if (
                module
                and metric_name.endswith("_score")
                and metric_name.replace("_score", "") in content_safety_metrics
            ):
                content_safety_cols.append(col)

    content_safety_df = df[content_safety_cols]
    defect_rates = {}
    for col in content_safety_df.columns:
        defect_rate_name = col.replace("_score", "_defect_rate")
        col_with_numeric_values = cast(List[float], pd.to_numeric(content_safety_df[col], errors="coerce"))
        try:
            col_with_boolean_values = apply_transform_nan_safe(
                col_with_numeric_values, lambda x: 1 if x >= CONTENT_SAFETY_DEFECT_RATE_THRESHOLD_DEFAULT else 0
            )
            defect_rates[defect_rate_name] = round(list_mean_nan_safe(col_with_boolean_values), 2)
        except EvaluationException:  # only exception that can be cause is all NaN values
            msg = f"All score evaluations are NaN/None for column {col}. No aggregation can be performed."
            LOGGER.warning(msg)

    return content_safety_cols, defect_rates


def _aggregate_label_defect_metrics(df: pd.DataFrame) -> Tuple[List[str], Dict[str, float]]:
    """Find and aggregate defect rates for label-based metrics. Returns both a list
    of columns that were used to calculate defect rates and the defect rates themselves.

    :param df: The dataframe of evaluation results.
    :type df: ~pandas.DataFrame
    :return: A tuple; the first element is a list of dataframe columns that were used to calculate defect rates,
        and the second element is a dictionary of defect column names and defect rates.
    :rtype: Tuple[List[str], Dict[str, float]]
    """
    handled_metrics = [
        EvaluationMetrics.PROTECTED_MATERIAL,
        EvaluationMetrics.FICTIONAL_CHARACTERS,
        EvaluationMetrics.ARTWORK,
        EvaluationMetrics.LOGOS_AND_BRANDS,
        _InternalEvaluationMetrics.ECI,
        EvaluationMetrics.XPIA,
        EvaluationMetrics.CODE_VULNERABILITY,
        EvaluationMetrics.UNGROUNDED_ATTRIBUTES,
    ]
    label_cols = []
    details_cols = []
    for col in df.columns:
        metric_name = col.split(".")[1]
        if metric_name.endswith("_label") and metric_name.replace("_label", "").lower() in handled_metrics:
            label_cols.append(col)
        if metric_name.endswith("_details") and metric_name.replace("_details", "").lower() in handled_metrics:
            details_cols = col

    label_df = df[label_cols]
    defect_rates = {}
    for col in label_df.columns:
        defect_rate_name = col.replace("_label", "_defect_rate")
        col_with_boolean_values = cast(List[float], pd.to_numeric(label_df[col], errors="coerce"))
        try:
            defect_rates[defect_rate_name] = round(list_mean_nan_safe(col_with_boolean_values), 2)
        except EvaluationException:  # only exception that can be cause is all NaN values
            msg = f"All score evaluations are NaN/None for column {col}. No aggregation can be performed."
            LOGGER.warning(msg)

    if details_cols:
        details_df = df[details_cols]
        detail_defect_rates = {}

        for key, value in details_df.items():
            _process_rows(value, detail_defect_rates)

        for key, value in detail_defect_rates.items():
            col_with_boolean_values = pd.to_numeric(value, errors="coerce")
            try:
                defect_rates[f"{details_cols}.{key}_defect_rate"] = round(
                    list_mean_nan_safe(col_with_boolean_values), 2
                )
            except EvaluationException:  # only exception that can be cause is all NaN values
                msg = f"All score evaluations are NaN/None for column {key}. No aggregation can be performed."
                LOGGER.warning(msg)

    return label_cols, defect_rates


def _process_rows(row, detail_defect_rates):
    for key, value in row.items():
        if key not in detail_defect_rates:
            detail_defect_rates[key] = []
        detail_defect_rates[key].append(value)
    return detail_defect_rates


def _aggregation_binary_output(df: pd.DataFrame) -> Dict[str, float]:
    """
    Aggregate binary output results (pass/fail) from evaluation dataframe.

    For each evaluator, calculates the proportion of "pass" results.

    :param df: The dataframe of evaluation results.
    :type df: ~pandas.DataFrame
    :return: A dictionary mapping evaluator names to the proportion of pass results.
    :rtype: Dict[str, float]
    """
    results = {}

    # Find all columns that end with "_result"
    result_columns = [col for col in df.columns if col.startswith("outputs.") and col.endswith("_result")]

    for col in result_columns:
        # Extract the evaluator name from the column name
        # (outputs.<evaluator>.<metric>_result)
        parts = col.split(".")
        evaluator_name = None
        if len(parts) >= 3:
            evaluator_name = parts[1]
        else:
            LOGGER.warning(
                "Skipping column '%s' due to unexpected format. Expected at least three parts separated by '.'", col
            )
            continue
        if evaluator_name:
            # Count the occurrences of each unique value (pass/fail)
            value_counts = df[col].value_counts().to_dict()

            # Calculate the proportion of EVALUATION_PASS_FAIL_MAPPING[True] results
            total_rows = len(df)
            pass_count = value_counts.get(EVALUATION_PASS_FAIL_MAPPING[True], 0)
            proportion = pass_count / total_rows if total_rows > 0 else 0.0

            # Set the result with the evaluator name as the key
            result_key = f"{evaluator_name}.{BINARY_AGGREGATE_SUFFIX}"
            results[result_key] = round(proportion, 2)

    return results


def _aggregate_metrics(df: pd.DataFrame, evaluators: Dict[str, Callable]) -> Dict[str, float]:
    """Aggregate metrics from the evaluation results.
    On top of naively calculating the mean of most metrics, this function also identifies certain columns
    that represent defect rates and renames them accordingly. Other columns in the dataframe are dropped.
    EX: protected_material_label -> protected_material_defect_rate

    :param df: The dataframe of evaluation results.
    :type df: ~pandas.DataFrame
    :param evaluators:  A dictionary mapping of strings to evaluator classes.
    :type evaluators: Dict[str, Callable]
    :return: The aggregated metrics.
    :rtype: Dict[str, float]
    """
    binary_metrics = _aggregation_binary_output(df)

    df.rename(columns={col: col.replace("outputs.", "") for col in df.columns}, inplace=True)

    handled_columns = []
    defect_rates = {}
    # Rename certain columns as defect rates if we know that's what their aggregates represent
    # Content safety metrics
    content_safety_cols, cs_defect_rates = _aggregate_content_safety_metrics(df, evaluators)
    other_renamed_cols, renamed_cols = _aggregate_other_metrics(df)
    handled_columns.extend(content_safety_cols)
    handled_columns.extend(other_renamed_cols)
    defect_rates.update(cs_defect_rates)
    defect_rates.update(renamed_cols)
    # Label-based (true/false) metrics where 'true' means 'something is wrong'
    label_cols, label_defect_rates = _aggregate_label_defect_metrics(df)
    handled_columns.extend(label_cols)
    defect_rates.update(label_defect_rates)

    # For rest of metrics, we will calculate mean
    df.drop(columns=handled_columns, inplace=True)

    # NOTE: nan/None values don't count as as booleans, so boolean columns with
    # nan/None values won't have a mean produced from them.
    # This is different from label-based known evaluators, which have special handling.
    mean_value = df.mean(numeric_only=True)
    metrics = mean_value.to_dict()
    # Add defect rates back into metrics
    metrics.update(defect_rates)

    # Add binary threshold metrics based on pass/fail results
    metrics.update(binary_metrics)

    return metrics


def _validate_columns_for_target(
    df: pd.DataFrame,
    target: Callable,
) -> None:
    """
    Check that all columns needed by target function are present.

    :param df: The data frame to be validated.
    :type df: pd.DataFrame
    :param target: The callable to be applied to data set.
    :type target: Optional[Callable]
    :raises EvaluationException: If the column starts with "__outputs." or if the input data contains missing fields.
    """
    if any(c.startswith(Prefixes.TSG_OUTPUTS) for c in df.columns):
        msg = "The column cannot start from " f'"{Prefixes.TSG_OUTPUTS}" if target was defined.'
        raise EvaluationException(
            message=msg,
            internal_message=msg,
            target=ErrorTarget.EVALUATE,
            category=ErrorCategory.INVALID_VALUE,
            blame=ErrorBlame.USER_ERROR,
        )
    # If the target function is given, it may return
    # several columns and hence we cannot check the availability of columns
    # without knowing target function semantics.
    # Instead, here we will validate the columns, taken by target.
    required_inputs = [
        param.name
        for param in inspect.signature(target).parameters.values()
        if param.default == inspect.Parameter.empty and param.name not in ["kwargs", "args", "self"]
    ]

    missing_inputs = [col for col in required_inputs if col not in df.columns]
    if missing_inputs:
        msg = f"Missing required inputs for target: {missing_inputs}."
        raise EvaluationException(
            message=msg,
            target=ErrorTarget.EVALUATE,
            category=ErrorCategory.MISSING_FIELD,
            blame=ErrorBlame.USER_ERROR,
        )

def _parse_columns_from_mapping(mapped_values) -> Set[str]:
    """
    Parse column names from mapping values like "${data.column}" or "${run.outputs.column}"
    
    :param mapped_values: Collection of mapping value strings
    :return: Set of actual column names that should exist in the dataframe
    """
    columns = set()
    for value in mapped_values:
        # Match patterns like ${data.column} or ${run.outputs.column}
        match = re.match(r'\$\{(?:data|run\.outputs)\.(.+)\}', value)
        if match:
            column_name = match.group(1)
            # If it's a run.outputs reference, it should be prefixed with __outputs.
            if value.startswith("${run.outputs."):
                column_name = f"{Prefixes.TSG_OUTPUTS}{column_name}"
            columns.add(column_name)
    return columns

def _validate_columns_for_evaluators(
    df: pd.DataFrame,
    evaluators: Dict[str, Callable],
    target: Optional[Callable],
    target_generated_columns: Optional[Set[str]],
    column_mapping: Dict[str, Dict[str, str]],
) -> None:
    """
    Check that all columns needed by evaluators are present.

    :param df: The data frame to be validated.
    :type df: pd.DataFrame
    :param evaluators: The dictionary of evaluators.
    :type evaluators: Dict[str, Callable]
    :param target: The callable to be applied to data set.
    :type target: Optional[Callable]
    :param target_generated_columns: The set of columns generated by the target callable.
    :type target_generated_columns: Optional[Set[str]]
    :param column_mapping: Dictionary mapping evaluator name to evaluator column mapping.
    :type column_mapping: Dict[str, Dict[str, str]]
    :raises EvaluationException: If data is missing required inputs or if the target callable did not generate the necessary columns.
    """
    evaluator_col_set = set()

    # Define system fields that should be ignored during validation
    IGNORED_SYSTEM_FIELDS = {"_id", "line_number"}

    # Add logging to understand the validation
    print("=== Column Validation ===")
    print(f"Columns needed by evaluators: {evaluator_col_set}")
    print(f"Available columns in dataframe: {list(df.columns)}")
    print(f"Column mapping: {json.dumps(column_mapping, indent=2)}")

    col_in_df_set = set(df.columns.tolist())

    # Apply column mapping and check all required columns are in the data frame
    for evaluator_name, evaluator in evaluators.items():
        # Handle evaluators with column_mapping
        if evaluator_name in column_mapping:
            mapped_values = column_mapping[evaluator_name].values()
            required_columns = _parse_columns_from_mapping(mapped_values)

            print(f"Evaluator '{evaluator_name}' requires columns: {required_columns}")

            missing_columns = required_columns - col_in_df_set
            if missing_columns:
                print(f"Missing columns for evaluator '{evaluator_name}': {missing_columns}")
                msg = f"Couldn't find these mapping relations: {', '.join(mapped_values)}." \
                      f" Please make sure your input mapping keys and values match your YAML input section and input data"
                raise EvaluationException(
                    message=msg,
                    internal_message=msg,
                    target=ErrorTarget.EVALUATE,
                    category=ErrorCategory.INVALID_VALUE,
                    blame=ErrorBlame.USER_ERROR,
                )

    missing_inputs_per_evaluator = {}

    for evaluator_name, evaluator in evaluators.items():
        # Apply column mapping
        mapping_config = column_mapping.get(evaluator_name, column_mapping.get("default", None))
        new_df = _apply_column_mapping(df, mapping_config)

        # Validate input data for evaluator
        is_built_in = evaluator.__module__.startswith("azure.ai.evaluation")
        if is_built_in:
            evaluator_params = [
                param.name
                for param in inspect.signature(evaluator).parameters.values()
                if param.name not in ["kwargs", "args", "self"]
            ]

            if "conversation" in evaluator_params and "conversation" in new_df.columns:
                missing_inputs = []
            else:
                optional_params = (
                    cast(Any, evaluator)._OPTIONAL_PARAMS
                    if hasattr(evaluator, "_OPTIONAL_PARAMS")
                    else []
                )
                excluded_params = set(new_df.columns).union(optional_params).union(IGNORED_SYSTEM_FIELDS)
                missing_inputs = [col for col in evaluator_params if col not in excluded_params]

                if "conversation" in missing_inputs:
                    if not (evaluator_params == ["conversation"] and missing_inputs == ["conversation"]):
                        missing_inputs.remove("conversation")
        else:
            evaluator_params = [
                param.name
                for param in inspect.signature(evaluator).parameters.values()
                if param.default == inspect.Parameter.empty and param.name not in ["kwargs", "args", "self"]
            ]

            # Filter out ignored system fields from missing inputs
            missing_inputs = [col for col in evaluator_params if col not in new_df.columns and col not in IGNORED_SYSTEM_FIELDS]

        if missing_inputs:
            missing_inputs_per_evaluator[evaluator_name] = missing_inputs

    if missing_inputs_per_evaluator:
        msg = "Some evaluators are missing required inputs:\n"
        for evaluator_name, missing in missing_inputs_per_evaluator.items():
            msg += f"\n Evaluator '{evaluator_name}' is missing required inputs: {missing}\n"

        msg += "\nTo resolve this issue:\n"
        msg += "- Ensure the data contains required inputs.\n"
        if target is not None:
            msg += "- Verify that the target is generating the necessary columns for the evaluators. "
            msg += f"Currently generated columns: {target_generated_columns} \n"
        msg += "- Check that the column mapping is correctly configured."

        raise EvaluationException(
            message=msg.strip(),
            target=ErrorTarget.EVALUATE,
            category=ErrorCategory.MISSING_FIELD,
            blame=ErrorBlame.USER_ERROR,
        )


def _validate_and_load_data(target, data, evaluators, output_path, azure_ai_project, evaluation_name):
    if data is None:
        msg = "The 'data' parameter is required for evaluation."
        raise EvaluationException(
            message=msg,
            target=ErrorTarget.EVALUATE,
            category=ErrorCategory.INVALID_VALUE,
            blame=ErrorBlame.USER_ERROR,
        )
    if not isinstance(data, (os.PathLike, str)):
        msg = "The 'data' parameter must be a string or a path-like object."
        raise EvaluationException(
            message=msg,
            target=ErrorTarget.EVALUATE,
            category=ErrorCategory.INVALID_VALUE,
            blame=ErrorBlame.USER_ERROR,
        )
    if not os.path.exists(data):
        msg = f"The input data file path '{data}' does not exist."
        raise EvaluationException(
            message=msg,
            target=ErrorTarget.EVALUATE,
            category=ErrorCategory.INVALID_VALUE,
            blame=ErrorBlame.USER_ERROR,
        )

    if target is not None:
        if not callable(target):
            msg = "The 'target' parameter must be a callable function."
            raise EvaluationException(
                message=msg,
                target=ErrorTarget.EVALUATE,
                category=ErrorCategory.INVALID_VALUE,
                blame=ErrorBlame.USER_ERROR,
            )

    if not evaluators:
        msg = "The 'evaluators' parameter is required and cannot be None or empty."
        raise EvaluationException(
            message=msg,
            target=ErrorTarget.EVALUATE,
            category=ErrorCategory.INVALID_VALUE,
            blame=ErrorBlame.USER_ERROR,
        )
    if not isinstance(evaluators, dict):
        msg = "The 'evaluators' parameter must be a dictionary."
        raise EvaluationException(
            message=msg,
            target=ErrorTarget.EVALUATE,
            category=ErrorCategory.INVALID_VALUE,
            blame=ErrorBlame.USER_ERROR,
        )

    if output_path is not None:
        if not isinstance(output_path, (os.PathLike, str)):
            msg = "The 'output_path' parameter must be a string or a path-like object."
            raise EvaluationException(
                message=msg,
                target=ErrorTarget.EVALUATE,
                category=ErrorCategory.INVALID_VALUE,
                blame=ErrorBlame.USER_ERROR,
            )

        output_dir = output_path if os.path.isdir(output_path) else os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            msg = f"The output directory '{output_dir}' does not exist. Please create the directory manually."
            raise EvaluationException(
                message=msg,
                target=ErrorTarget.EVALUATE,
                category=ErrorCategory.INVALID_VALUE,
                blame=ErrorBlame.USER_ERROR,
            )

    if azure_ai_project is not None:
        validate_azure_ai_project(azure_ai_project)

    if evaluation_name is not None:
        if not isinstance(evaluation_name, str) or not evaluation_name.strip():
            msg = "The 'evaluation_name' parameter must be a non-empty string."
            raise EvaluationException(
                message=msg,
                target=ErrorTarget.EVALUATE,
                category=ErrorCategory.INVALID_VALUE,
                blame=ErrorBlame.USER_ERROR,
            )

    try:
        data_loader = DataLoaderFactory.get_loader(data)
        initial_data_df = data_loader.load()
    except Exception as e:
        raise EvaluationException(
            message=f"Unable to load data from '{data}'. Supported formats are JSONL and CSV. Detailed error: {e}.",
            target=ErrorTarget.EVALUATE,
            category=ErrorCategory.INVALID_VALUE,
            blame=ErrorBlame.USER_ERROR,
        ) from e

    return initial_data_df


def _apply_target_to_data(
    target: Callable,
    data: Union[str, os.PathLike, pd.DataFrame],
    batch_client: BatchClient,
    initial_data: pd.DataFrame,
    evaluation_name: Optional[str] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Set[str], BatchClientRun]:
    """
    Apply the target function to the data set and return updated data and generated columns.

    :param target: The function to be applied to data.
    :type target: Callable
    :param data: The path to input jsonl or csv file.
    :type data: Union[str, os.PathLike]
    :param batch_client: The promptflow client to be used.
    :type batch_client: PFClient
    :param initial_data: The data frame with the loaded data.
    :type initial_data: pd.DataFrame
    :param evaluation_name: The name of the evaluation.
    :type evaluation_name: Optional[str]
    :return: The tuple, containing data frame and the list of added columns.
    :rtype: Tuple[pandas.DataFrame, List[str]]
    """

    _run_name = kwargs.get("_run_name")
    with TargetRunContext(batch_client):
        run: BatchClientRun = batch_client.run(
            flow=target,
            display_name=evaluation_name,
            data=data,
            stream=True,
            name=_run_name,
            evaluator_name=getattr(target, "__qualname__", "TARGET"),
        )
        target_output: pd.DataFrame = batch_client.get_details(run, all_results=True)
        run_summary = batch_client.get_run_summary(run)
    
    LOGGER.info("=== Target Output Analysis ===")
    LOGGER.info(f"All columns in target output: {list(target_output.columns)}")
    output_columns = [col for col in target_output.columns if col.startswith(Prefixes.OUTPUTS)]
    LOGGER.info(f"Output columns from target: {output_columns}")
    LOGGER.info(f"Output column names (without prefix): {[col[len(Prefixes.OUTPUTS):] for col in output_columns]}")
    
    # Log a sample of the data to see what's actually there
    if len(target_output) > 0:
        LOGGER.info(f"Sample target output (first row): {target_output.iloc[0].to_dict()}")

    if run_summary["completed_lines"] == 0:
        msg = (
            f"Evaluation target failed to produce any results."
            f" Please check the logs at {run_summary['log_path']} for more details about cause of failure."
        )
        raise EvaluationException(
            message=msg,
            target=ErrorTarget.EVALUATE,
            category=ErrorCategory.FAILED_EXECUTION,
            blame=ErrorBlame.USER_ERROR,
        )

    # Log a warning if some rows failed
    failed_lines = run_summary.get("failed_lines", 0)
    completed_lines = run_summary["completed_lines"]
    total_lines = failed_lines + completed_lines

    if failed_lines > 0:
        LOGGER.warning(
            f"Target function completed {completed_lines} out of {total_lines} rows. "
            f"{failed_lines} rows failed and will be filled with NaN values."
        )

    # Remove input and output prefix
    generated_columns = {
        col[len(Prefixes.OUTPUTS) :] for col in target_output.columns if col.startswith(Prefixes.OUTPUTS)
    }
    # Sort output by line numbers
    target_output.set_index(f"inputs.{LINE_NUMBER}", inplace=True)
    target_output.sort_index(inplace=True)

    initial_data_with_line_numbers = initial_data.copy()
    initial_data_with_line_numbers[LINE_NUMBER] = range(len(initial_data))

    complete_index = initial_data_with_line_numbers[LINE_NUMBER]
    target_output = target_output.reindex(complete_index)

    target_output.reset_index(inplace=True, drop=False)
    # target_output contains only input columns, taken by function,
    # so we need to concatenate it to the input data frame.
    drop_columns = list(filter(lambda x: x.startswith("inputs"), target_output.columns))
    target_output.drop(drop_columns, inplace=True, axis=1)
    # Rename outputs columns to __outputs
    rename_dict = {col: col.replace(Prefixes.OUTPUTS, Prefixes.TSG_OUTPUTS) for col in target_output.columns}
    target_output.rename(columns=rename_dict, inplace=True)
    # Concatenate output to input - now both dataframes have the same number of rows
    target_output = pd.concat([initial_data, target_output], axis=1)

    return target_output, generated_columns, run


def _process_column_mappings(
    column_mapping: Dict[str, Optional[Dict[str, str]]],
) -> Dict[str, Dict[str, str]]:
    """Process column_mapping to replace ${target.} with ${data.}

    :param column_mapping: The configuration for evaluators.
    :type column_mapping: Dict[str, Optional[Dict[str, str]]]
    :return: The processed configuration.
    :rtype: Dict[str, Dict[str, str]]
    """

    processed_config: Dict[str, Dict[str, str]] = {}

    expected_references = re.compile(r"^\$\{(target|data)\.([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*)\}$")

    # Add logging here
    LOGGER.info("Processing column mappings...")
    LOGGER.info(f"Input column_mapping: {json.dumps(column_mapping, indent=2)}")

    if column_mapping:
        for evaluator, mapping_config in column_mapping.items():
            if isinstance(mapping_config, dict):
                processed_config[evaluator] = {}

                for map_to_key, map_value in mapping_config.items():
                    # Check if there's any unexpected reference other than ${target.} or ${data.}
                    if not expected_references.search(map_value):
                        msg = "Unexpected references detected in 'column_mapping'. Ensure only ${target.} and ${data.} are used."
                        raise EvaluationException(
                            message=msg,
                            internal_message=msg,
                            target=ErrorTarget.EVALUATE,
                            category=ErrorCategory.INVALID_VALUE,
                            blame=ErrorBlame.USER_ERROR,
                        )

                    # Replace ${target.} with ${run.outputs.}
                    original_value = map_value
                    processed_config[evaluator][map_to_key] = map_value.replace("${target.", "${run.outputs.")
                    
                    # Add logging for each mapping
                    if original_value != processed_config[evaluator][map_to_key]:
                        LOGGER.info(f"  Evaluator '{evaluator}': Converted '{map_to_key}': '{original_value}' -> '{processed_config[evaluator][map_to_key]}'")
                    else:
                        LOGGER.info(f"  Evaluator '{evaluator}': Keeping '{map_to_key}': '{map_value}'")

    LOGGER.info(f"Processed column_mapping: {json.dumps(processed_config, indent=2)}")
    return processed_config


def evaluate(
    *,
    data: Union[str, os.PathLike],
    evaluators: Dict[str, Union[Callable, AzureOpenAIGrader]],
    evaluation_name: Optional[str] = None,
    target: Optional[Callable] = None,
    evaluator_config: Optional[Dict[str, EvaluatorConfig]] = None,
    azure_ai_project: Optional[Union[str, AzureAIProject]] = None,
    output_path: Optional[Union[str, os.PathLike]] = None,
    fail_on_evaluator_errors: bool = False,
    **kwargs,
) -> EvaluationResult:
    """Evaluates target or data with built-in or custom evaluators. If both target and data are provided,
        data will be run through target function and then results will be evaluated.

    :keyword data: Path to the data to be evaluated or passed to target if target is set.
        JSONL and CSV files are supported.  `target` and `data` both cannot be None. Required.
    :paramtype data: str
    :keyword evaluators: Evaluators to be used for evaluation. It should be a dictionary with key as alias for evaluator
        and value as the evaluator function. Also accepts AzureOpenAIGrader instances as values, which are processed separately.
        Required.
    :paramtype evaluators: Dict[str, Union[Callable, ~azure.ai.evaluation.AzureOpenAIGrader]]
    :keyword evaluation_name: Display name of the evaluation.
    :paramtype evaluation_name: Optional[str]
    :keyword target: Target to be evaluated. `target` and `data` both cannot be None
    :paramtype target: Optional[Callable]
    :keyword evaluator_config: Configuration for evaluators. The configuration should be a dictionary with evaluator
        names as keys and a values that are dictionaries containing the column mappings. The column mappings should
        be a dictionary with keys as the column names in the evaluator input and values as the column names in the
        input data or data generated by target.
    :paramtype evaluator_config: Optional[Dict[str, ~azure.ai.evaluation.EvaluatorConfig]]
    :keyword output_path: The local folder or file path to save evaluation results to if set. If folder path is provided
          the results will be saved to a file named `evaluation_results.json` in the folder.
    :paramtype output_path: Optional[str]
    :keyword azure_ai_project: The Azure AI project, which can either be a string representing the project endpoint
        or an instance of AzureAIProject. It contains subscription id, resource group, and project name.
    :paramtype azure_ai_project: Optional[Union[str, ~azure.ai.evaluation.AzureAIProject]]
    :keyword fail_on_evaluator_errors: Whether or not the evaluation should cancel early with an EvaluationException
        if ANY evaluator fails during their evaluation.
        Defaults to false, which means that evaluations will continue regardless of failures.
        If such failures occur, metrics may be missing, and evidence of failures can be found in the evaluation's logs.
    :paramtype fail_on_evaluator_errors: bool
    :keyword user_agent: A string to append to the default user-agent sent with evaluation http requests
    :paramtype user_agent: Optional[str]
    :return: Evaluation results.
    :rtype: ~azure.ai.evaluation.EvaluationResult

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START evaluate_method]
            :end-before: [END evaluate_method]
            :language: python
            :dedent: 8
            :caption: Run an evaluation on local data with one or more evaluators using azure.ai.evaluation.AzureAIProject

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START evaluate_method]
            :end-before: [END evaluate_method]
            :language: python
            :dedent: 8
            :caption: Run an evaluation on local data with one or more evaluators using Azure AI Project URL in following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}
    """
    try:
        user_agent: Optional[str] = kwargs.get("user_agent")
        with UserAgentSingleton().add_useragent_product(user_agent) if user_agent else contextlib.nullcontext():
            return _evaluate(
                evaluation_name=evaluation_name,
                target=target,
                data=data,
                evaluators_and_graders=evaluators,
                evaluator_config=evaluator_config,
                azure_ai_project=azure_ai_project,
                output_path=output_path,
                fail_on_evaluator_errors=fail_on_evaluator_errors,
                **kwargs,
            )
    except Exception as e:
        # Handle multiprocess bootstrap error
        bootstrap_error = (
            "An attempt has been made to start a new process before the\n        "
            "current process has finished its bootstrapping phase."
        )
        if bootstrap_error in str(e):
            error_message = (
                "The evaluation failed due to an error during multiprocess bootstrapping."
                "Please ensure the evaluate API is properly guarded with the '__main__' block:\n\n"
                "    if __name__ == '__main__':\n"
                "        evaluate(...)"
            )
            raise EvaluationException(
                message=error_message,
                internal_message=error_message,
                target=ErrorTarget.EVALUATE,
                category=ErrorCategory.FAILED_EXECUTION,
                blame=ErrorBlame.USER_ERROR,
            ) from e

        # Ensure a consistent user experience when encountering errors by converting
        # all other exceptions to EvaluationException.
        if not isinstance(e, EvaluationException):
            raise EvaluationException(
                message=str(e),
                target=ErrorTarget.EVALUATE,
                category=ErrorCategory.FAILED_EXECUTION,
                blame=ErrorBlame.SYSTEM_ERROR,
            ) from e

        raise e


def _print_summary(per_evaluator_results: Dict[str, Any]) -> None:
    # Extract evaluators with a non-empty "run_summary"
    output_dict = {
        name: result["run_summary"] for name, result in per_evaluator_results.items() if result.get("run_summary")
    }

    if output_dict:
        print("======= Combined Run Summary (Per Evaluator) =======\n")
        print(json.dumps(output_dict, indent=4))
        print("\n====================================================\n")


def _print_fail_flag_warning() -> None:
    print(
        "Notice: fail_on_evaluator_errors is enabled. It is recommended that you disable "
        + "this flag for evaluations on large datasets (loosely defined as more than 10 rows of inputs, "
        + "or more than 4 evaluators). Using this flag on large datasets runs the risk of large runs failing "
        + "without producing any outputs, since a single failure will cancel the entire run "
        "when fail_on_evaluator_errors is enabled."
    )


def _evaluate(  # pylint: disable=too-many-locals,too-many-statements
    *,
    evaluators_and_graders: Dict[str, Union[Callable, AzureOpenAIGrader]],
    evaluation_name: Optional[str] = None,
    target: Optional[Callable] = None,
    data: Union[str, os.PathLike],
    evaluator_config: Optional[Dict[str, EvaluatorConfig]] = None,
    azure_ai_project: Optional[Union[str, AzureAIProject]] = None,
    output_path: Optional[Union[str, os.PathLike]] = None,
    fail_on_evaluator_errors: bool = False,
    **kwargs,
) -> EvaluationResult:
    if fail_on_evaluator_errors:
        _print_fail_flag_warning()

    # Turn inputted mess of data into a dataframe, apply targets if needed
    # split graders and evaluators, and verify that column mappings are sensible.
    validated_data = _preprocess_data(
        data=data,
        evaluators_and_graders=evaluators_and_graders,
        evaluator_config=evaluator_config,
        target=target,
        output_path=output_path,
        azure_ai_project=azure_ai_project,
        evaluation_name=evaluation_name,
        **kwargs,
    )

    # extract relevant info from validated data
    column_mapping = validated_data["column_mapping"]
    evaluators = validated_data["evaluators"]
    graders = validated_data["graders"]
    input_data_df = validated_data["input_data_df"]
    results_df = pd.DataFrame()
    metrics: Dict[str, float] = {}
    eval_run_info_list: List[OAIEvalRunCreationInfo] = []

    # Start OAI eval runs if any graders are present.
    need_oai_run = len(graders) > 0
    need_local_run = len(evaluators) > 0
    need_get_oai_results = False
    got_local_results = False
    if need_oai_run:
        try:
            aoi_name = evaluation_name if evaluation_name else DEFAULT_OAI_EVAL_RUN_NAME
            eval_run_info_list = _begin_aoai_evaluation(graders, column_mapping, input_data_df, aoi_name)
            need_get_oai_results = len(eval_run_info_list) > 0
        except EvaluationException as e:
            if need_local_run:
                # If there are normal evaluators, don't stop execution and try to run
                # those.
                LOGGER.warning(
                    "Remote Azure Open AI grader evaluations failed during run creation."
                    + " Continuing with local evaluators."
                )
                LOGGER.warning(e)
            else:
                raise e

    # Evaluate 'normal' evaluators. This includes built-in evaluators and any user-supplied callables.
    if need_local_run:
        try:
            eval_result_df, eval_metrics, per_evaluator_results = _run_callable_evaluators(
                validated_data=validated_data, fail_on_evaluator_errors=fail_on_evaluator_errors
            )
            results_df = eval_result_df
            metrics = eval_metrics
            got_local_results = True
            # TODO figure out how to update this printing to include OAI results?
            _print_summary(per_evaluator_results)
        except EvaluationException as e:
            if need_get_oai_results:
                # If there are OAI graders, we only print a warning on local failures.
                LOGGER.warning("Local evaluations failed. Will still attempt to retrieve online grader results.")
                LOGGER.warning(e)
            else:
                raise e

    # Retrieve OAI eval run results if needed.
    if need_get_oai_results:
        try:
            aoai_results, aoai_metrics = _get_evaluation_run_results(eval_run_info_list)  # type: ignore
            # Post build TODO: add equivalent of  _print_summary(per_evaluator_results) here

            # Combine results if both evaluators and graders are present
            if len(evaluators) > 0:
                results_df = pd.concat([results_df, aoai_results], axis=1)
                metrics.update(aoai_metrics)
            else:
                # Otherwise combine aoai results with input data df to include input columns in outputs.
                results_df = pd.concat([input_data_df, aoai_results], axis=1)
                metrics = aoai_metrics
        except EvaluationException as e:
            if got_local_results:
                # If there are local eval results, we only print a warning on OAI failure.
                LOGGER.warning("Remote Azure Open AI grader evaluations failed. Still returning local results.")
                LOGGER.warning(e)
            else:
                raise e

    # Done with all evaluations, message outputs into final forms, and log results if needed.
    name_map = _map_names_to_builtins(evaluators, graders)
    if is_onedp_project(azure_ai_project):
        studio_url = _log_metrics_and_instance_results_onedp(
            metrics, results_df, azure_ai_project, evaluation_name, name_map, **kwargs
        )
    else:
        # Since tracing is disabled, pass None for target_run so a dummy evaluation run will be created each time.
        trace_destination = _trace_destination_from_project_scope(azure_ai_project) if azure_ai_project else None
        studio_url = None
        if trace_destination:
            studio_url = _log_metrics_and_instance_results(
                metrics, results_df, trace_destination, None, evaluation_name, name_map, **kwargs
            )

    result_df_dict = results_df.to_dict("records")
    result: EvaluationResult = {"rows": result_df_dict, "metrics": metrics, "studio_url": studio_url}  # type: ignore

    if output_path:
        _write_output(output_path, result)

    return result


def _preprocess_data(
    data: Union[str, os.PathLike],
    evaluators_and_graders: Dict[str, Union[Callable, AzureOpenAIGrader]],
    evaluator_config: Optional[Dict[str, EvaluatorConfig]] = None,
    target: Optional[Callable] = None,
    output_path: Optional[Union[str, os.PathLike]] = None,
    azure_ai_project: Optional[Union[str, AzureAIProject]] = None,
    evaluation_name: Optional[str] = None,
    **kwargs,
) -> __ValidatedData:
    # Process evaluator config to replace ${target.} with ${data.}
    if evaluator_config is None:
        evaluator_config = {}

    input_data_df = _validate_and_load_data(
        target, data, evaluators_and_graders, output_path, azure_ai_project, evaluation_name
    )
    if target is not None:
        _validate_columns_for_target(input_data_df, target)

    # extract column mapping dicts into dictionary mapping evaluator name to column mapping
    column_mapping = _process_column_mappings(
        {
            evaluator_name: evaluator_configuration.get("column_mapping", None)
            for evaluator_name, evaluator_configuration in evaluator_config.items()
        }
    )

    # Create default configuration for evaluators that directly maps
    # input data names to keyword inputs of the same name in the evaluators.
    column_mapping = column_mapping or {}
    column_mapping.setdefault("default", {})

    # Split normal evaluators and OAI graders
    evaluators, graders = _split_evaluators_and_grader_configs(evaluators_and_graders)

    target_run: Optional[BatchClientRun] = None
    target_generated_columns: Set[str] = set()
    batch_run_client: BatchClient
    batch_run_data: Union[str, os.PathLike, pd.DataFrame] = data

    if kwargs.pop("_use_run_submitter_client", False):
        batch_run_client = RunSubmitterClient()
        batch_run_data = input_data_df
    elif kwargs.pop("_use_pf_client", True):
        batch_run_client = ProxyClient(user_agent=UserAgentSingleton().value)
        # Ensure the absolute path is passed to pf.run, as relative path doesn't work with
        # multiple evaluators. If the path is already absolute, abspath will return the original path.
        batch_run_data = os.path.abspath(data)
    else:
        batch_run_client = CodeClient()
        batch_run_data = input_data_df

    # If target is set, apply 1-1 column mapping from target outputs to evaluator inputs
    if data is not None and target is not None:
        LOGGER.info(f"Target function detected. Applying target to data...")
        input_data_df, target_generated_columns, target_run = _apply_target_to_data(
            target, batch_run_data, batch_run_client, input_data_df, evaluation_name, **kwargs
        )
        
        LOGGER.info(f"Target generated columns: {target_generated_columns}")
        LOGGER.info(f"Columns in data after target application: {list(input_data_df.columns)}")

        # Now we need to update the column mappings to use data references instead of run.outputs references
        # since we're creating a complete dataset that includes failed rows
        for evaluator_name, mapping in column_mapping.items():
            updated_mapping = {}
            for key, value in mapping.items():
                # If the value references run.outputs (which came from ${target.x}), 
                # convert it to reference the data column instead
                if value.startswith("${run.outputs."):
                    # Extract the column name from ${run.outputs.column_name}
                    match = re.match(r'\$\{run\.outputs\.(.+)\}', value)
                    if match:
                        col_name = match.group(1)
                        # Convert to data reference with __outputs prefix
                        updated_mapping[key] = f"${{data.{Prefixes.TSG_OUTPUTS}{col_name}}}"
                        LOGGER.info(f"Updated mapping for '{evaluator_name}': '{key}' from '{value}' to '{updated_mapping[key]}'")
                else:
                    updated_mapping[key] = value
            column_mapping[evaluator_name] = updated_mapping

        # This ensures that evaluators get all rows (including failed ones with NaN values)
        if isinstance(batch_run_client, ProxyClient):
            # Create a temporary JSONL file with the complete dataframe
            temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
            try:
                # Create a copy of the dataframe and remove the line_number column to avoid duplication
                temp_df = input_data_df.copy()
                if LINE_NUMBER in temp_df.columns:
                    temp_df = temp_df.drop(columns=[LINE_NUMBER])
                    LOGGER.info(f"Removed {LINE_NUMBER} column from temporary dataframe before writing to file")
                
                for _, row in temp_df.iterrows():
                    row_dict = row.to_dict()
                    temp_file.write(json.dumps(row_dict) + "\n")
                temp_file.close()
                batch_run_data = temp_file.name
                LOGGER.info(f"Created temporary JSONL file with {len(temp_df)} rows at {temp_file.name}")

                # Update column mappings to use data references instead of run outputs
                for evaluator_name, mapping in column_mapping.items():
                    mapped_to_values = set(mapping.values())
                    for col in target_generated_columns:
                        # Use data reference instead of run output to ensure we get all rows
                        target_reference = f"${{data.{Prefixes.TSG_OUTPUTS}{col}}}"

                        # We will add our mapping only if customer did not map target output.
                        if col not in mapping and target_reference not in mapped_to_values:
                            column_mapping[evaluator_name][col] = target_reference

                # Don't pass the target_run since we're now using the complete dataframe
                target_run = None

            except Exception as e:
                # Clean up the temp file if something goes wrong
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                raise e
        else:
            # For DataFrame-based clients, update batch_run_data to use the updated input_data_df
            batch_run_data = input_data_df

            # Update column mappings for DataFrame clients
            for evaluator_name, mapping in column_mapping.items():
                mapped_to_values = set(mapping.values())
                for col in target_generated_columns:
                    target_reference = f"${{data.{Prefixes.TSG_OUTPUTS}{col}}}"

                    # We will add our mapping only if customer did not map target output.
                    if col not in mapping and target_reference not in mapped_to_values:
                        column_mapping[evaluator_name][col] = target_reference
                        LOGGER.info(f"Auto-mapping target output '{col}' to '{target_reference}' for evaluator '{evaluator_name}'")

    # After we have generated all columns, we can check if we have everything we need for evaluators.
    LOGGER.info(f"Final column mapping before validation: {json.dumps(column_mapping, indent=2)}")
    _validate_columns_for_evaluators(input_data_df, evaluators, target, target_generated_columns, column_mapping)

    # Apply 1-1 mapping from input data to evaluator inputs, excluding values already assigned
    # via target mapping.
    # If both the data and the output dictionary of the target function
    # have the same column, then the target function value is used.
    if input_data_df is not None:
        for col in input_data_df.columns:
            # Ignore columns added by target mapping. These are formatted as "__outputs.<column_name>"
            # Also ignore columns that are already in config, since they've been covered by target mapping.
            if not col.startswith(Prefixes.TSG_OUTPUTS) and col not in column_mapping["default"].keys():
                column_mapping["default"][col] = f"${{data.{col}}}"
                LOGGER.info(f"Auto-mapping input column '{col}' to '${{data.{col}}}' in default mapping")

    LOGGER.info(f"Final complete column mapping: {json.dumps(column_mapping, indent=2)}")

    return __ValidatedData(
        evaluators=evaluators,
        graders=graders,
        input_data_df=input_data_df,
        column_mapping=column_mapping,
        target_run=target_run,
        batch_run_client=batch_run_client,
        batch_run_data=batch_run_data,
    )


def _rename_columns_conditionally(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change the column names for data frame. The change happens inplace.

    The columns with _OUTPUTS prefix will not be changed. _OUTPUTS prefix will
    will be added to columns in target_generated set. The rest columns will get
    ".inputs" prefix.

    :param df: The data frame to apply changes to.
    :type df: pandas.DataFrame
    :return: The changed data frame.
    :rtype: pandas.DataFrame
    """
    rename_dict = {}
    for col in df.columns:
        # Skip the line_number column - it will be handled separately
        if col == LINE_NUMBER:
            continue
        # Rename columns generated by target.
        elif Prefixes.TSG_OUTPUTS in col:
            rename_dict[col] = col.replace(Prefixes.TSG_OUTPUTS, Prefixes.OUTPUTS)
        else:
            rename_dict[col] = f"inputs.{col}"
    
    df.rename(columns=rename_dict, inplace=True)
    
    # Remove the line_number column if it exists to avoid duplication issues
    if LINE_NUMBER in df.columns:
        df.drop(columns=[LINE_NUMBER], inplace=True)
        LOGGER.info(f"Dropped {LINE_NUMBER} column during column renaming")
    
    return df

def _run_callable_evaluators(
    validated_data: __ValidatedData,
    fail_on_evaluator_errors: bool = False,
    **kwargs,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, __EvaluatorInfo]]:
    """Run callable evaluators on the input data, handling missing columns gracefully."""
    
    # Extract needed values
    batch_run_client = validated_data["batch_run_client"]
    target_run = validated_data["target_run"]
    batch_run_data = validated_data["batch_run_data"]
    column_mapping = validated_data["column_mapping"]
    evaluators = validated_data["evaluators"]
    input_data_df = validated_data["input_data_df"]

    # Store temporary files for cleanup
    temp_files_to_cleanup = []
    
    try:
        with EvalRunContext(batch_run_client):
            runs = {}
            per_evaluator_results: Dict[str, __EvaluatorInfo] = {}
            
            for evaluator_name, evaluator in evaluators.items():
                # Apply column mapping to get the data that this evaluator needs
                mapping_config = column_mapping.get(evaluator_name, column_mapping.get("default", None))
                mapped_df = _apply_column_mapping(input_data_df, mapping_config)
                
                # Check for rows with missing required columns
                evaluator_params = [
                    param.name
                    for param in inspect.signature(evaluator).parameters.values()
                    if param.default == inspect.Parameter.empty and param.name not in ["kwargs", "args", "self"]
                ]
                
                # Create a mask for rows that have all required columns (not NaN)
                valid_rows_mask = pd.Series([True] * len(mapped_df))
                for param in evaluator_params:
                    if param in mapped_df.columns:
                        valid_rows_mask &= mapped_df[param].notna()
                
                skipped_rows = (~valid_rows_mask).sum()
                if skipped_rows > 0:
                    LOGGER.warning(
                        f"Evaluator '{evaluator_name}' will skip {skipped_rows} rows due to missing required columns"
                    )
                
                # Prepare data for this evaluator
                if valid_rows_mask.all():
                    # All rows are valid, use original data
                    evaluator_data = batch_run_data
                else:
                    # Some rows have missing values, we need to filter
                    if isinstance(batch_run_client, ProxyClient):
                        # For ProxyClient, create a temporary JSONL file with only valid rows
                        valid_df = input_data_df[valid_rows_mask].copy()
                        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
                        temp_files_to_cleanup.append(temp_file.name)
                        
                        # Remove line_number column to avoid duplication
                        if LINE_NUMBER in valid_df.columns:
                            valid_df = valid_df.drop(columns=[LINE_NUMBER])
                        
                        for _, row in valid_df.iterrows():
                            row_dict = row.to_dict()
                            temp_file.write(json.dumps(row_dict) + "\n")
                        temp_file.close()
                        evaluator_data = temp_file.name
                        
                        LOGGER.info(f"Created temporary JSONL file for evaluator '{evaluator_name}' "
                                  f"with {len(valid_df)} valid rows out of {len(input_data_df)} total rows")
                    else:
                        # For DataFrame-based clients, filter the dataframe
                        evaluator_data = input_data_df[valid_rows_mask].copy()
                
                if valid_rows_mask.any():
                    # Run evaluator only if there are valid rows
                    run = batch_run_client.run(
                        flow=evaluator,
                        data=evaluator_data,
                        run=None,  # Don't pass target_run since we're handling row filtering ourselves
                        evaluator_name=evaluator_name,
                        column_mapping=mapping_config,
                        stream=True,
                        name=kwargs.get("_run_name"),
                    )
                    runs[evaluator_name] = run
                    
                    # Get results
                    result_df = batch_run_client.get_details(run, all_results=True)
                    metrics = batch_run_client.get_metrics(run)
                    run_summary = batch_run_client.get_run_summary(run)
                    
                    # Create a full result DataFrame with NaN for skipped rows
                    full_result_df = pd.DataFrame(index=range(len(input_data_df)))
                    
                    # Get output columns from the evaluator results (excluding input columns)
                    output_cols = [col for col in result_df.columns if not col.startswith("inputs.")]
                    for col in output_cols:
                        full_result_df[col] = pd.NA
                    
                    # Fill in the results for valid rows
                    valid_indices = input_data_df[valid_rows_mask].index
                    for i, col in enumerate(output_cols):
                        full_result_df.loc[valid_indices, col] = result_df[col].values
                    
                    per_evaluator_results[evaluator_name] = {
                        "result": full_result_df,
                        "metrics": metrics,
                        "run_summary": run_summary,
                    }
                else:
                    # No valid rows for this evaluator
                    LOGGER.warning(f"Evaluator '{evaluator_name}' skipped all rows due to missing required columns")
                    
                    # Create empty result with NaN values
                    full_result_df = pd.DataFrame(index=range(len(input_data_df)))
                    
                    # Add empty columns based on what the evaluator would typically output
                    # This is a best-effort approach - we add a result column at minimum
                    full_result_df[f"{evaluator_name}_result"] = pd.NA
                    
                    per_evaluator_results[evaluator_name] = {
                        "result": full_result_df,
                        "metrics": {},
                        "run_summary": {
                            "completed_lines": 0,
                            "failed_lines": 0,
                            "skipped_lines": len(input_data_df)
                        },
                    }
                    
    finally:
        # Clean up all temporary files
        for temp_file in temp_files_to_cleanup:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    LOGGER.info(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    LOGGER.warning(f"Failed to clean up temporary file {temp_file}: {e}")
    
    # Concatenate all results
    evaluators_result_df = pd.DataFrame()
    evaluators_metric = {}
    
    for evaluator_name, evaluator_result in per_evaluator_results.items():
        if fail_on_evaluator_errors and evaluator_result["run_summary"].get("failed_lines", 0) > 0:
            _print_summary(per_evaluator_results)
            if "log_path" in evaluator_result["run_summary"]:
                _turn_error_logs_into_exception(evaluator_result["run_summary"]["log_path"] + "/error.json")

        evaluator_result_df = evaluator_result["result"]

        # Drop input columns if they exist
        input_cols_to_drop = [col for col in evaluator_result_df.columns if str(col).startswith(Prefixes.INPUTS)]
        if input_cols_to_drop:
            evaluator_result_df = evaluator_result_df.drop(columns=input_cols_to_drop)

        # Rename output columns
        evaluator_result_df.rename(
            columns={
                col: f"outputs.{evaluator_name}.{str(col).replace(Prefixes.OUTPUTS, '')}"
                for col in evaluator_result_df.columns
            },
            inplace=True,
        )

        evaluators_result_df = (
            pd.concat([evaluators_result_df, evaluator_result_df], axis=1, verify_integrity=True)
            if not evaluators_result_df.empty
            else evaluator_result_df
        )

        evaluators_metric.update({f"{evaluator_name}.{k}": v for k, v in evaluator_result["metrics"].items()})

    # Rename columns generated by target function
    input_data_df = _rename_columns_conditionally(validated_data["input_data_df"])
    
    # Concatenate input data with evaluator results
    if evaluators_result_df.empty:
        eval_result_df = input_data_df
    else:
        eval_result_df = pd.concat([input_data_df, evaluators_result_df], axis=1, verify_integrity=True)
    
    # Aggregate metrics (this will handle NaN values appropriately)
    eval_metrics = _aggregate_metrics(evaluators_result_df, evaluators)
    eval_metrics.update(evaluators_metric)

    return eval_result_df, eval_metrics, per_evaluator_results


def _map_names_to_builtins(
    evaluators: Dict[str, Callable],
    graders: Dict[str, AzureOpenAIGrader],
) -> Dict[str, str]:
    """
    Construct a mapping from user-supplied evaluator names to which known, built-in
    evaluator or grader they refer to. Custom evaluators are excluded from the mapping
    as we only want to track built-in evaluators and graders.

    :param evaluators: The dictionary of evaluators.
    :type evaluators: Dict[str, Callable]
    :param graders: The dictionary of graders.
    :type graders: Dict[str, AzureOpenAIGrader]
    :param evaluator_config: The configuration for evaluators.
    :type evaluator_config: Optional[Dict[str, EvaluatorConfig]]

    """
    from .._eval_mapping import EVAL_CLASS_MAP

    name_map = {}

    for name, evaluator in evaluators.items():
        # Check if the evaluator is a known built-in evaluator
        found_eval = False
        for eval_class, eval_id in EVAL_CLASS_MAP.items():
            if isinstance(evaluator, eval_class):
                name_map[name] = eval_id
                found_eval = True
                break
        if not found_eval:
            # Skip custom evaluators - we only want to track built-in evaluators
            pass

    for name, grader in graders.items():
        name_map[name] = grader.id

    return name_map


def _turn_error_logs_into_exception(log_path: str) -> None:
    """Produce an EvaluationException using the contents of the inputted
    file as the error message.

    :param log_path: The path to the error log file.
    :type log_path: str
    """
    with open(log_path, "r", encoding=DefaultOpenEncoding.READ) as file:
        error_message = file.read()
    raise EvaluationException(
        message=error_message,
        target=ErrorTarget.EVALUATE,
        category=ErrorCategory.FAILED_EXECUTION,
        blame=ErrorBlame.UNKNOWN,
    )
