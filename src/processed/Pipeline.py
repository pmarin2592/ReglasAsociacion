import pandas as pd
from src.processed.AssociationPipeline import AssociationPipeline


class Pipeline:
    def __init__(self, df: pd.DataFrame, min_support: float, min_confidence: float):
        self.df = df
        self.min_support = min_support
        self.min_confidence = min_confidence

    def _run_laptops(self, df: pd.DataFrame):
        pipeline = AssociationPipeline(
            df=df,
            combined_columns=[
                "Color", "Type", "Suitable For",
                "Processor Brand", "Processor Name",
                "RAM", "RAM Type", "Operating System"
            ],
            transaction_columns=["Sales Package"],
            columns_to_keep=[
                "name", "Sales Package", "Color", "Type", "Suitable For",
                "Processor Brand", "Processor Name", "RAM", "RAM Type",
                "Operating System",
            ],
            min_support=self.min_support,
            min_confidence=self.min_confidence,
            output_dir="outputs/laptops",
            run_apriori=True,
            run_eclat=True,
            cat_columns_to_plot=[
                "Processor Brand", "RAM Type", "Operating System", "Type"
            ],
        )
        return pipeline.run(), "outputs/laptops"

    def _run_tourism(self, df: pd.DataFrame):
        pipeline = AssociationPipeline(
            df=df,
            transaction_columns=["Interests", "Sites Visited"],
            columns_to_keep=[
                "Tourist ID", "Age", "Interests", "Sites Visited",
                "Preferred Tour Duration", "Tour Duration",
            ],
            min_support=self.min_support,
            min_confidence=self.min_confidence,
            output_dir="outputs/tourism",
            run_apriori=True,
            run_eclat=True,
            cat_columns_to_plot=["Interests"],
            separator=",",
        )
        return pipeline.run(), "outputs/tourism"

    def execute(self, dataSet: str):
        if dataSet == "laptops":
            return self._run_laptops(self.df)
        elif dataSet == "tourism":
            return self._run_tourism(self.df)
        return {}, ""