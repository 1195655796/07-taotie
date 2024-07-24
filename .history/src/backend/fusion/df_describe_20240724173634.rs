use arrow::{
    array::{ArrayRef, RecordBatch, StringArray},
    compute::{cast, concat},
    datatypes::{DataType, Field, Schema, SchemaRef},
};

use datafusion::logical_expr::approx_percentile_cont;
use datafusion::{
    dataframe::DataFrame,
    logical_expr::{avg, case, col, count, is_null, lit, max, median, min, stddev, sum},
};
use std::sync::Arc;


#[allow(unused)]
pub struct DescribeDataFrame {
    df: DataFrame,
    functions: &'static [&'static str],
    schema: SchemaRef,
}

#[allow(unused)]
impl DescribeDataFrame {
    pub fn new(df: DataFrame) -> Self {
        let functions = &[
            "count",
            "null_count",
            "mean",
            "std",
            "min",
            "max",
            "median",
            "percentile25",
            "percentile50",
            "percentile75",
        ];

        let original_schema_fields = df.schema().fields().iter();

        //define describe column
        let mut describe_schemas = vec![Field::new("describe", DataType::Utf8, false)];
        describe_schemas.extend(original_schema_fields.clone().map(|field| {
            if field.data_type().is_numeric() {
                Field::new(field.name(), DataType::Float64, true)
            } else {
                Field::new(field.name(), DataType::Utf8, true)
            }
        }));
        Self {
            df,
            functions,
            schema: Arc::new(Schema::new(describe_schemas)),
        }
    }

    pub async fn to_record_batch(&self) -> anyhow::Result<RecordBatch> {
        let original_schema_fields = self.df.schema().fields().iter();

        let batches = vec![
            self.count(),
            self.null_count(),
            self.mean(),
            self.stddev(),
            self.min(),
            self.max(),
            self.medium(),
            self.percentile(25),
            self.percentile(50),
            self.percentile(75),
        ];

        // first column with function names
        let mut describe_col_vec: Vec<ArrayRef> = vec![Arc::new(StringArray::from(
            self.functions
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>(),
        ))];
        for field in original_schema_fields {
            let mut array_data = vec![];
            for result in batches.iter() {
                let array_ref = match result {
                    Ok(df) => {
                        let batchs = df.clone().collect().await;
                        match batchs {
                            Ok(batchs)
                                if batchs.len() == 1
                                    && batchs[0].column_by_name(field.name()).is_some() =>
                            {
                                let column = batchs[0].column_by_name(field.name()).unwrap();
                                if field.data_type().is_numeric() {
                                    cast(column, &DataType::Float64)?
                                } else {
                                    cast(column, &DataType::Utf8)?
                                }
                            }
                            _ => Arc::new(StringArray::from(vec!["null"])),
                        }
                    }
                    //Handling error when only boolean/binary column, and in other cases
                    Err(err)
                        if err.to_string().contains(
                            "Error during planning: \
                                          Aggregate requires at least one grouping \
                                          or aggregate expression",
                        ) =>
                    {
                        Arc::new(StringArray::from(vec!["null"]))
                    }
                    Err(other_err) => {
                        panic!("{other_err}")
                    }
                };
                array_data.push(array_ref);
            }
            describe_col_vec.push(concat(
                array_data
                    .iter()
                    .map(|af| af.as_ref())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )?);
        }

        let batch = RecordBatch::try_new(self.schema.clone(), describe_col_vec)?;

        Ok(batch)
    }

    fn count(&self) -> anyhow::Result<DataFrame> {
        let original_schema_fields = self.df.schema().fields().iter();
        let ret = self.df.clone().aggregate(
            vec![],
            original_schema_fields
                .clone()
                .map(|f| count(col(f.name())).alias(f.name()))
                .collect::<Vec<_>>(),
        )?;
        Ok(ret)
    }

    fn null_count(&self) -> anyhow::Result<DataFrame> {
        let original_schema_fields = self.df.schema().fields().iter();
        let ret = self.df.clone().aggregate(
            vec![],
            original_schema_fields
                .clone()
                .map(|f| {
                    sum(case(is_null(col(f.name())))
                        .when(lit(true), lit(1))
                        .otherwise(lit(0))
                        .unwrap())
                    .alias(f.name())
                })
                .collect::<Vec<_>>(),
        )?;
        Ok(ret)
    }

    fn mean(&self) -> anyhow::Result<DataFrame> {
        let original_schema_fields = self.df.schema().fields().iter();
        let ret = self.df.clone().aggregate(
            vec![],
            original_schema_fields
                .clone()
                .filter(|f| f.data_type().is_numeric())
                .map(|f| avg(col(f.name())).alias(f.name()))
                .collect::<Vec<_>>(),
        )?;
        Ok(ret)
    }

    fn stddev(&self) -> anyhow::Result<DataFrame> {
        let original_schema_fields = self.df.schema().fields().iter();
        let ret = self.df.clone().aggregate(
            vec![],
            original_schema_fields
                .clone()
                .filter(|f| f.data_type().is_numeric())
                .map(|f| stddev(col(f.name())).alias(f.name()))
                .collect::<Vec<_>>(),
        )?;
        Ok(ret)
    }

    fn min(&self) -> anyhow::Result<DataFrame> {
        let original_schema_fields = self.df.schema().fields().iter();
        let ret = self.df.clone().aggregate(
            vec![],
            original_schema_fields
                .clone()
                .filter(|f| !matches!(f.data_type(), DataType::Binary | DataType::Boolean))
                .map(|f| min(col(f.name())).alias(f.name()))
                .collect::<Vec<_>>(),
        )?;
        Ok(ret)
    }

    fn max(&self) -> anyhow::Result<DataFrame> {
        let original_schema_fields = self.df.schema().fields().iter();
        let ret = self.df.clone().aggregate(
            vec![],
            original_schema_fields
                .clone()
                .filter(|f| !matches!(f.data_type(), DataType::Binary | DataType::Boolean))
                .map(|f| max(col(f.name())).alias(f.name()))
                .collect::<Vec<_>>(),
        )?;
        Ok(ret)
    }

    fn medium(&self) -> anyhow::Result<DataFrame> {
        let original_schema_fields = self.df.schema().fields().iter();
        let ret = self.df.clone().aggregate(
            vec![],
            original_schema_fields
                .clone()
                .filter(|f| f.data_type().is_numeric())
                .map(|f| median(col(f.name())).alias(f.name()))
                .collect::<Vec<_>>(),
        )?;
        Ok(ret)
    }
    pub fn percentile(&self, percentile: u8) -> anyhow::Result<DataFrame> {
        let original_schema_fields = self.df.schema().fields().iter();
        let ret = self.df.clone().aggregate(
            vec![],
            original_schema_fields
                .clone()
                .filter(|f| f.data_type().is_numeric())
                .map(|f| {
                    approx_percentile_cont(lit(percentile as f64 / 100.0), col(f.name()))
                        .alias(f.name())
                })
                .collect::<Vec<_>>(),
        )?;
        Ok(ret)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::*;
    use arrow::record_batch::RecordBatch;
    use arrow::array::{Float64Array, Int32Array, StringArray, ArrayRef};
    use std::sync::Arc;

    fn create_test_dataframe() -> DataFrame {
        // Create a simple RecordBatch
        let schema = Arc::new(Schema::new(vec![
            Field::new("float_col", DataType::Float64, false),
            Field::new("int_col", DataType::Int32, false),
            Field::new("string_col", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])) as ArrayRef,
                Arc::new(Int32Array::from(vec![4, 5, 6])) as ArrayRef,
                Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
            ],
        ).unwrap();

        // Convert the RecordBatch to a DataFrame
        let ctx = SessionContext::new();
        ctx.read_batch(batch).unwrap()
    }

    #[tokio::test]
    async fn test_try_new() {
        let df = create_test_dataframe();
        let describer = DataFrameDescriber::try_new(df);
        assert!(describer.is_ok());
    }

    #[tokio::test]
    async fn test_describe_total() {
        let df = create_test_dataframe();
        let describer = DataFrameDescriber::try_new(df).unwrap();

        let described_df = describer.describe().await;
        assert!(described_df.is_ok());

        let result = described_df.unwrap().collect().await.unwrap();
        assert_eq!(result.len(), 1);  // should have one row for each statistic
        assert_eq!(result[0].num_columns(), 4);  // describe + three columns
    }

    #[tokio::test]
    async fn test_describe_mean() {
        let df = create_test_dataframe();
        let describer = DataFrameDescriber::try_new(df).unwrap();

        let described_df = describer.describe().await;
        assert!(described_df.is_ok());

        let result = described_df.unwrap().collect().await.unwrap();
        let mean_value = result[0].column(1).as_any().downcast_ref::<Float64Array>().unwrap().value(0);
        assert_eq!(mean_value, 2.0);  // mean of [1.0, 2.0, 3.0]
    }

    #[tokio::test]
    async fn test_describe_percentiles() {
        let df = create_test_dataframe();
        let describer = DataFrameDescriber::try_new(df).unwrap();

        let described_df = describer.describe().await;
        assert!(described_df.is_ok());

        let result = described_df.unwrap().collect().await.unwrap();
        let percentile_25_value = result[0].column(1).as_any().downcast_ref::<Float64Array>().unwrap().value(7);  // assuming percentile_25 is the 7th column
        assert_eq!(percentile_25_value, 1.5);  // 25th percentile of [1.0, 2.0, 3.0]
    }
}