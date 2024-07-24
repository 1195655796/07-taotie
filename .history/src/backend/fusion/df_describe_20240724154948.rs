use arrow::{
    array::{ArrayRef, RecordBatch, StringArray},
    compute::{cast, concat},
    datatypes::{DataType, Field, Schema, SchemaRef},
};

use datafusion::logical_expr::Expr;
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
            self.percentile25(),
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

    async fn calculate_quantile(df: &DataFrame, quantile: f64) -> Result<DataFrame> {
        let original_schema_fields = df.schema().fields().iter();
        let quantile_exprs: Vec<_> = original_schema_fields
            .filter(|f| matches!(f.data_type(), DataType::Float64 | DataType::Int64))
            .map(|f| {
                let col_name = f.name().clone();
                async move {
                    // 对数据进行排序
                    let sorted_df = df.sort(vec![col(&col_name)], vec![true]).await?;
                    // 计算索引
                    let idx = (quantile * sorted_df.num_rows() as f64).round() as usize;
                    // 提取相应的值
                    let quantile_value = sorted_df.column(&col_name).unwrap().get(idx).unwrap();
                    Ok(quantile_value)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // 构建结果 DataFrame
        let schema = df.schema().clone();
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            quantile_exprs
                .into_iter()
                .map(|val| Arc::new(ScalarValue::new(val)))
                .collect(),
        )?;
        Ok(DataFrame::new(vec![batch]))
    }

    fn percentile0_25(&self) -> Result<DataFrame> {
        Self::calculate_quantile(&self.df, 0.25).await
    }
}
