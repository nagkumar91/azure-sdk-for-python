# Refactoring Plan for Red Team Module

This document outlines a detailed plan for refactoring the `_red_team.py` file into smaller, more focused components to improve code maintainability and readability.

## Current Issues

The `_red_team.py` file is approximately 2000+ lines of code with numerous responsibilities:
- Retry mechanism management
- Orchestration of attack strategies
- Evaluation of conversations
- MLFlow logging and integration
- Result processing and formatting
- Attack strategy management
- PyRIT integration

This large file size causes several problems:
1. Reduced code readability
2. Difficulty in maintaining and updating functionality
3. Challenges in unit testing specific components
4. Limited reusability of individual components

## Proposed Directory Structure

```
azure/ai/evaluation/red_team/
├── __init__.py                  (Exports main RedTeam class and public types)
├── _red_team.py                 (Main class with significantly reduced size)
├── _retry/
│   ├── __init__.py
│   └── retry_manager.py         (Retry logic and configuration)
├── _orchestration/
│   ├── __init__.py
│   └── orchestrator_manager.py  (Orchestration and prompt handling)
├── _evaluation/
│   ├── __init__.py
│   └── evaluation_processor.py  (Evaluation logic)
├── _mlflow/
│   ├── __init__.py
│   └── mlflow_logger.py         (MLFlow interaction)
├── _result_processing/
│   ├── __init__.py
│   └── result_processor.py      (Result compilation and formatting)
├── _utils/                      (Existing utility directory)
│   ├── __init__.py
│   ├── constants.py             (Moved from existing _utils module) 
│   ├── formatting_utils.py      (Expanded utility module)
│   ├── logging_utils.py         (Moved from existing _utils module)
│   ├── metric_mapping.py        (Moved from existing _utils module)
│   └── strategy_utils.py        (Moved from existing _utils module)
```

## Component Details

### 1. RetryManager (`_retry/retry_manager.py`)

**Purpose:** Encapsulate all retry-related functionality to provide a consistent retry mechanism.

**Responsibilities:**
- Define retry constants (MAX_RETRY_ATTEMPTS, etc.)
- Create retry configurations for different types of operations
- Log retry attempts and errors
- Provide utility methods for retry predicates

**Key Methods:**
- `create_retry_config()`
- `log_retry_attempt()`
- `log_retry_error()`

**Class Definition:**
```python
class RetryManager:
    def __init__(self, logger, max_retry_attempts=5, min_retry_wait=2, max_retry_wait=30):
        self.logger = logger
        self.max_retry_attempts = max_retry_attempts
        self.min_retry_wait_seconds = min_retry_wait
        self.max_retry_wait_seconds = max_retry_wait

    def create_retry_config(self):
        # Create and return retry configuration dictionary

    def log_retry_attempt(self, retry_state):
        # Log retry attempt information

    def log_retry_error(self, retry_state):
        # Log final error when retries are exhausted
```

### 2. OrchestratorManager (`_orchestration/orchestrator_manager.py`)

**Purpose:** Handle all aspects of orchestration, including prompt processing, conversion, and PyRIT interactions.

**Responsibilities:**
- Create and configure orchestrators
- Apply prompt converters to inputs
- Manage conversation execution
- Write outputs to files
- Handle orchestrator failures and timeouts

**Key Methods:**
- `prompt_sending_orchestrator()`
- `write_pyrit_outputs_to_file()`
- `get_orchestrators_for_attack_strategies()`
- `get_converter_for_strategy()`
- `get_chat_target()`

**Class Definition:**
```python
class OrchestratorManager:
    def __init__(self, logger, retry_manager, red_team_info, scan_output_dir=None):
        self.logger = logger
        self.retry_manager = retry_manager
        self.red_team_info = red_team_info
        self.scan_output_dir = scan_output_dir
        self.task_statuses = {}

    async def prompt_sending_orchestrator(self, chat_target, all_prompts, converter, strategy_name="unknown", risk_category="unknown", timeout=120):
        # Create and configure PyRIT orchestrator
        
    def write_pyrit_outputs_to_file(self, orchestrator, strategy_name, risk_category, batch_idx=None):
        # Extract and write PyRIT outputs to file
        
    def get_orchestrators_for_attack_strategies(self, attack_strategy):
        # Return appropriate orchestrators for strategies
        
    def get_converter_for_strategy(self, attack_strategy):
        # Get converter(s) for a strategy
        
    def get_chat_target(self, target):
        # Convert different target types to PromptChatTarget
```

### 3. EvaluationProcessor (`_evaluation/evaluation_processor.py`)

**Purpose:** Handle all evaluation logic, including running evaluations, processing conversation data, and interfacing with the RAI service.

**Responsibilities:**
- Evaluate conversations for risk categories
- Get attack objectives for evaluation
- Process evaluation results
- Handle evaluation errors and retries

**Key Methods:**
- `evaluate()`
- `evaluate_conversation()`
- `get_attack_objectives()`

**Class Definition:**
```python
class EvaluationProcessor:
    def __init__(self, logger, retry_manager, rai_client, generated_rai_client, credential, azure_ai_project, red_team_info=None):
        self.logger = logger
        self.retry_manager = retry_manager
        self.rai_client = rai_client
        self.generated_rai_client = generated_rai_client
        self.credential = credential
        self.azure_ai_project = azure_ai_project
        self.red_team_info = red_team_info or {}
        self.attack_objectives = {}

    async def evaluate(self, data_path, risk_category, strategy, scan_name=None, output_path=None, skip_evals=False):
        # Evaluate data against risk category using appropriate metrics
        
    async def evaluate_conversation(self, conversation, metric_name, strategy_name, risk_category, idx):
        # Evaluate a single conversation
        
    async def get_attack_objectives(self, risk_category=None, application_scenario=None, strategy=None):
        # Get attack objectives from RAI service or custom dataset
```

### 4. MLFlowLogger (`_mlflow/mlflow_logger.py`)

**Purpose:** Handle all interactions with MLFlow for logging evaluation results and managing runs.

**Responsibilities:**
- Start MLFlow runs
- Log metrics and artifacts
- Generate URLs for Azure AI Studio
- Handle logging errors and retries

**Key Methods:**
- `start_redteam_mlflow_run()`
- `log_redteam_results_to_mlflow()`

**Class Definition:**
```python
class MLFlowLogger:
    def __init__(self, logger, retry_manager, azure_ai_project, credential, scan_output_dir=None):
        self.logger = logger
        self.retry_manager = retry_manager
        self.azure_ai_project = azure_ai_project
        self.credential = credential
        self.scan_output_dir = scan_output_dir
        self.trace_destination = None

    def start_redteam_mlflow_run(self, run_name=None):
        # Start and configure an MLFlow run
        
    async def log_redteam_results_to_mlflow(self, redteam_output, eval_run, skip_evals=False):
        # Log results, metrics, and artifacts to MLFlow
```

### 5. ResultProcessor (`_result_processing/result_processor.py`)

**Purpose:** Handle all aspects of result generation, formatting, and aggregation.

**Responsibilities:**
- Convert tracking data to RedTeamResult format
- Generate scorecard from results
- Calculate metrics and statistics
- Format results for reporting

**Key Methods:**
- `to_red_team_result()`
- `to_scorecard()`
- `get_attack_success()`

**Class Definition:**
```python
class ResultProcessor:
    def __init__(self, logger, red_team_info, risk_categories, application_scenario=None, scan_output_dir=None):
        self.logger = logger
        self.red_team_info = red_team_info
        self.risk_categories = risk_categories
        self.application_scenario = application_scenario
        self.scan_output_dir = scan_output_dir

    def to_red_team_result(self):
        # Convert red_team_info to RedTeamResult
        
    def to_scorecard(self, redteam_result):
        # Generate human-readable scorecard
        
    def get_attack_success(self, result):
        # Determine if an attack was successful
```

## Main RedTeam Class Modifications

The main `RedTeam` class will be significantly simplified by delegating responsibilities to the component classes:

```python
@experimental
class RedTeam():
    """
    This class uses various attack strategies to test the robustness of AI models against adversarial inputs.
    It logs the results of these evaluations and provides detailed scorecards summarizing the attack success rates.
    
    :param azure_ai_project: The Azure AI project configuration
    :type azure_ai_project: dict
    :param credential: The credential to authenticate with Azure services
    :type credential: TokenCredential
    :param risk_categories: List of risk categories to generate attack objectives for (optional if custom_attack_seed_prompts is provided)
    :type risk_categories: Optional[List[RiskCategory]]
    :param num_objectives: Number of objectives to generate per risk category
    :type num_objectives: int
    :param application_scenario: Description of the application scenario for context
    :type application_scenario: Optional[str]
    :param custom_attack_seed_prompts: Path to a JSON file containing custom attack seed prompts (can be absolute or relative path)
    :type custom_attack_seed_prompts: Optional[str]
    :param output_dir: Directory to save output files (optional)
    :type output_dir: Optional[str]
    """
    def __init__(
            self,
            azure_ai_project,
            credential,
            *,
            risk_categories: Optional[List[RiskCategory]] = None,
            num_objectives: int = 10,
            application_scenario: Optional[str] = None,
            custom_attack_seed_prompts: Optional[str] = None,
            output_dir=None
        ):
        # Initialize core attributes
        self.azure_ai_project = validate_azure_ai_project(azure_ai_project)
        self.credential = credential
        self.output_dir = output_dir
        self.logger = setup_logger()
        
        # Initialize tracking variables
        self.task_statuses = {}
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = None
        self.scan_id = None
        self.scan_output_dir = None
        
        # Create component classes
        self.retry_manager = RetryManager(self.logger)
        self.orchestrator_manager = OrchestratorManager(self.logger, self.retry_manager, {})
        self.mlflow_logger = MLFlowLogger(self.logger, self.retry_manager, self.azure_ai_project, self.credential)
        
        # Initialize RAI clients
        self.token_manager = ManagedIdentityAPITokenManager(
            token_scope=TokenScope.DEFAULT_AZURE_MANAGEMENT,
            logger=logging.getLogger("RedTeamLogger"),
            credential=cast(TokenCredential, credential),
        )
        self.rai_client = RAIClient(azure_ai_project=self.azure_ai_project, token_manager=self.token_manager)
        self.generated_rai_client = GeneratedRAIClient(azure_ai_project=self.azure_ai_project, token_manager=self.token_manager.get_aad_credential())
        
        # Initialize evaluation processor after RAI clients
        self.evaluation_processor = EvaluationProcessor(
            self.logger, 
            self.retry_manager,
            self.rai_client,
            self.generated_rai_client,
            self.credential,
            self.azure_ai_project
        )
        
        # Initialize result processor (will be fully setup during scan)
        self.result_processor = None
        
        # Initialize attack objective generator
        self.attack_objective_generator = _AttackObjectiveGenerator(
            risk_categories=risk_categories, 
            num_objectives=num_objectives, 
            application_scenario=application_scenario, 
            custom_attack_seed_prompts=custom_attack_seed_prompts
        )
        
        # Store key parameters for later use
        self.risk_categories = risk_categories
        self.application_scenario = application_scenario
        
        # For tracking data and results
        self.red_team_info = {}
        self.attack_objectives = {}
        
        # Initialize PyRIT
        initialize_pyrit(memory_db_type=DUCK_DB)
        
        self.logger.debug("RedTeam initialized successfully")

    async def scan(self, target, **kwargs):
        # Main scanning method that orchestrates the components
        # Will be refactored to delegate to the component classes
```

## Implementation Approach

The refactoring will be implemented in the following steps:

1. **RetryManager Implementation** (1 day)
   - Create the `_retry` directory and implement the `RetryManager` class
   - Update the main class to use the `RetryManager`
   - Write tests for retry functionality

2. **OrchestratorManager Implementation** (2 days)
   - Create the `_orchestration` directory and implement the `OrchestratorManager` class
   - Move PyRIT interaction code from the main class
   - Update the main class to use the `OrchestratorManager`
   - Write tests for orchestration functionality

3. **EvaluationProcessor Implementation** (2 days)
   - Create the `_evaluation` directory and implement the `EvaluationProcessor` class
   - Move evaluation code and attack objectives logic
   - Update the main class to use the `EvaluationProcessor`
   - Write tests for evaluation functionality

4. **MLFlowLogger Implementation** (1 day)
   - Create the `_mlflow` directory and implement the `MLFlowLogger` class
   - Move MLFlow interaction code
   - Update the main class to use the `MLFlowLogger`
   - Write tests for logging functionality

5. **ResultProcessor Implementation** (2 days)
   - Create the `_result_processing` directory and implement the `ResultProcessor` class
   - Move result processing and formatting code
   - Update the main class to use the `ResultProcessor`
   - Write tests for result processing functionality

6. **Main Class Refactoring** (2 days)
   - Update the `_red_team.py` file to use all the new components
   - Ensure that all public interfaces remain compatible
   - Verify that all existing functionality continues to work

7. **Documentation and Testing** (2 days)
   - Update docstrings for all new classes and methods
   - Ensure comprehensive test coverage for all components
   - Verify end-to-end functionality

## Benefits of the New Structure

1. **Improved Maintainability**
   - Each component has a clearly defined responsibility
   - Changes to one component won't affect others
   - Easier to locate specific functionality

2. **Better Testability**
   - Components can be tested in isolation
   - Mocking dependencies is simpler
   - Test coverage will be easier to achieve

3. **Enhanced Readability**
   - Smaller, focused files are easier to understand
   - Clear separation of concerns between components
   - Consistent naming and organization

4. **Future Extensibility**
   - New attack strategies can be added without affecting evaluation
   - Alternative backends (e.g., non-MLFlow loggers) could be implemented
   - Components can be extended independently

## Backward Compatibility

The refactoring preserves the public API of the `RedTeam` class, so existing code will continue to work without modification. The internal implementation details will change, but this should be transparent to users of the class.

## Risk Mitigation

1. **Comprehensive Testing**
   - Each component will have unit tests
   - The refactored `RedTeam` class will have integration tests
   - End-to-end tests will verify overall functionality

2. **Gradual Implementation**
   - Components will be implemented and integrated one at a time
   - Each integration step will be tested before proceeding
   - Rollback plans will be in place for each step

3. **Documentation**
   - All component APIs will be fully documented
   - Implementation details will be documented where necessary
   - Comments will explain complex logic and design decisions

## Conclusion

This refactoring will transform a monolithic 2000+ line file into a modular, maintainable system of components with clear responsibilities. The effort required (approximately 2 weeks) will be well worth the improvements in code quality, maintainability, and future extensibility.
