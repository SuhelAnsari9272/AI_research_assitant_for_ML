import pandas as pd

from schemas.dataset_profile import DatasetProfile
from schemas.dataset_profile import GeneralProfile
from schemas.dataset_profile import StatisticsProfile
from schemas.dataset_profile import NumericalColumnStatistics
from schemas.dataset_profile import QualityProfile
from schemas.dataset_profile import MissingValueInfo
from schemas.dataset_profile import HighCardinalityColumn
from schemas.dataset_profile import ClassificationTargetProfile
from schemas.dataset_profile import RegressionTargetProfile
from schemas.project_config import ProjectConfig


def analyse_general(dataset: pd.DataFrame) -> GeneralProfile:
    numerical_columns = (
        dataset.select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    categorical_columns = (
        dataset.select_dtypes(include=["object", "category"])
        .columns
        .tolist()
    )

    datetime_columns = (
        dataset.select_dtypes(include=["datetime", "datetimetz"])
        .columns
        .tolist()
    )

    return GeneralProfile(
        n_rows=dataset.shape[0],
        n_cols =dataset.shape[1],
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
    )

def stats_analyse(dataset:pd.DataFrame, column : str) -> NumericalColumnStatistics:
    
    return NumericalColumnStatistics(
        column_name =column,
        mean = dataset[column].mean(),
        median = dataset[column].quantile(0.50),
        std = dataset[column].std(),
        minimum = dataset[column].min(),
        maximum = dataset[column].max(),
        q1 = dataset[column].quantile(0.25),
        q3 = dataset[column].quantile(0.75),
        skewness = dataset[column].skew(),
        kurtosis = dataset[column].kurt()
    )

def analyse_statistics(dataset:pd.DataFrame) -> StatisticsProfile :
    
    statistics_profiles = [] 

    # here we  need to filter for numerical columns

    for column in dataset.columns:
        if (dataset[column].dtype == 'int64') | (dataset[column].dtype == 'float64')  :
                statistics_profiles.append(stats_analyse(dataset, column))

    return StatisticsProfile(
        numerical_statistics=statistics_profiles
    )

def analyse_missing_info(dataset:pd.DataFrame) -> list[MissingValueInfo] :
     
     missing_columns = []
     for column in dataset.columns :
        missing_columns.append(
             MissingValueInfo(
            column_name=column,
            missing_count= dataset[column].isnull().sum(),
            missing_percentage=round(dataset[column].isnull().mean() * 100, 2) 
            )
        )
     
     return missing_columns

def analyse_constant_columns(dataset : pd.DataFrame)  -> list[str]:
    constant_cols = []

    for column in dataset.columns :

        unique_count = dataset[column].nunique(dropna=True)
        
        if unique_count == 1 :
            constant_cols.append(column)

    return constant_cols

def analyse_unique_identifier_columns(dataset : pd.DataFrame)  -> list[str]:
    unique_cols = []

    for column in dataset.columns :

        unique_count = dataset[column].nunique(dropna=True)
        
        if unique_count == dataset.shape[0] :
            unique_cols.append(column)

    return unique_cols

def analyse_high_cardinality_columns(dataset : pd.DataFrame)  -> list[HighCardinalityColumn]:
    high_cardinality_cols = []

    for column in dataset.columns :
        unique_count = dataset[column].nunique(dropna=True)
        
        if unique_count >= dataset.shape[0] //2  :
            
            high_cardinality_cols.append(HighCardinalityColumn(
                column_name=column,
                unique_count= unique_count, 
                unique_percentage= round((unique_count /dataset.shape[0]) * 100, 2)
            ))

    return high_cardinality_cols

def analyse_mostly_empty_columns(dataset : pd.DataFrame) -> list[str]:
    mostly_empty_cols = []

    for column in dataset.columns :
        missing_count =  dataset[column].isnull().sum()

        if missing_count >= dataset.shape[0] //2  :
            mostly_empty_cols.append(column)

    return mostly_empty_cols

def analyse_quality(dataset : pd.DataFrame) -> QualityProfile :
     
    duplicate_n_rows = len(dataset[dataset.duplicated(keep=False)])
    duplicate_percentage = round((duplicate_n_rows / dataset.shape[0])* 100 , 2)

    missing_columns = analyse_missing_info(dataset)
    constant_columns = analyse_constant_columns(dataset)
    unique_identifier_columns = analyse_unique_identifier_columns(dataset)
    high_cardinality_columns = analyse_high_cardinality_columns(dataset)

    mostly_empty_columns = analyse_mostly_empty_columns(dataset)

    def analyse_warnings() :
        warnings = []
    
        if  duplicate_n_rows > 0  or duplicate_percentage > 0: 
            warnings.append(f"Found {duplicate_n_rows} in the data.")
            warnings.append(f"Almost {duplicate_percentage} % of rows are duplicate")

        if missing_columns :
            warnings.append("Found some missing columns details :")
            for info in missing_columns :
                warnings.append(f"  - Column : {info.column_name} | Missing Count :  {info.missing_count} | Missing % :  {info.missing_percentage}")
        
        if constant_columns :
            warnings.append(f"Found {len(constant_columns)} constant columns : {constant_columns}")

        if unique_identifier_columns :
            warnings.append(f"Found {len(unique_identifier_columns)} Unique Identifier Columns : {unique_identifier_columns}")

        if high_cardinality_columns :
            warnings.append(f"Found {len(high_cardinality_columns)} High Cardinality Columns ")

            for column in high_cardinality_columns :
                warnings.append(f" - Column : {column.column_name} | unique_count : {column.unique_count} | unique_percentage : {column.unique_percentage}")

        if mostly_empty_columns :
            warnings.append(f"Found {len(mostly_empty_columns)} mostly empty columns . Columns : {mostly_empty_columns}")

        return warnings

    warnings = analyse_warnings()

    return QualityProfile(
        duplicate_rows = duplicate_n_rows,
        duplicate_percentage= duplicate_percentage,
        missing_columns=missing_columns,
        constant_columns=constant_columns,
        unique_identifier_columns=unique_identifier_columns,
        high_cardinality_columns=high_cardinality_columns,
        mostly_empty_columns= mostly_empty_columns,
        warnings=warnings
    )

def analyse_classification_target_profile(dataset, target_column) :

    class_counts = dataset[target_column].value_counts().sort_values(ascending=False)

    classes = list(class_counts.index.astype(str))
    class_distribution = class_counts.to_dict()
    imbalance_ratio = class_counts.max() / class_counts.min()
    majority_class = str(class_counts.idxmax())
    minority_class = str(class_counts.idxmin())
    
    return ClassificationTargetProfile(
        target_column=target_column, 
        problem_type="Classification",
        classes= classes,
        class_distribution= class_distribution,
        imbalance_ratio=imbalance_ratio,
        majority_class=majority_class,
        minority_class=minority_class
    )

def analyse_regression_target_profile(dataset, target_column) :
    
    mean = dataset[target_column].mean(),
    median = dataset[target_column].quantile(0.50),
    std = dataset[target_column].std(),
    minimum = dataset[target_column].min(),
    maximum = dataset[target_column].max(),
    # q1 = dataset[target_column].quantile(0.25),
    # q3 = dataset[target_column].quantile(0.75),
    skewness = dataset[target_column].skew(),
    # kurtosis = dataset[target_column].kurt()
    
    return RegressionTargetProfile(
        target_column=target_column,
        problem_type="Regression",
        mean = mean,
        median = median, 
        std= std,
        minimum= minimum, 
        maximum= maximum,
        skewness= skewness
    )

def analyse_target(dataset :pd.DataFrame, project_config : ProjectConfig):

    target_column = project_config.target_column
    problem_type =  project_config.problem_type

    if problem_type.lower() == "classification" :
        target_profile = analyse_classification_target_profile(dataset, target_column)


    elif problem_type.lower() == "regression" : 
        target_profile =  analyse_regression_target_profile(dataset, target_column)

    else :
        target_profile = None

    return target_profile

def analyse_dataset(dataset: pd.DataFrame, project_config : ProjectConfig) -> DatasetProfile:

    general =  analyse_general(dataset)
    statistics = analyse_statistics(dataset)
    quality = analyse_quality(dataset)
    target = analyse_target(dataset, project_config)

    return DatasetProfile(
        general= general, 
        statistics = statistics, 
        quality= quality,
        target=target
    )
    
