# Databricks notebook source
# CCLF File 1 (Part A Institutional Claims) Transformation to Staging Tables
# Author: amazingashis
# Date: 2025-09-02

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, to_date, current_timestamp, current_date, expr, udf
from pyspark.sql.types import StringType, DecimalType, IntegerType, DateType, TimestampType
import uuid
from datetime import datetime

# Configuration
input_path = "/mnt/data/raw/cclf/file1/"  # Update with your actual path
output_path = "/mnt/data/warehouse/staging/"  # Update with your actual path
file_date = "20250901"  # Update with your actual file date
run_date = "2025-09-02"  # Current date

# Create UDF for UUID generation
@udf(returnType=StringType())
def generate_uuid():
    return str(uuid.uuid4())

# COMMAND ----------

# Read CCLF File 1 data
print(f"Reading CCLF File 1 data from {input_path}")

try:
    # Adjust file format and options according to your actual input file
    df_file1 = spark.read.option("header", "true") \
                         .option("inferSchema", "false") \
                         .option("delimiter", ",") \
                         .csv(input_path)
    
    print(f"Successfully read {df_file1.count()} records from CCLF File 1")
    
except Exception as e:
    print(f"Error reading CCLF File 1: {str(e)}")
    raise

# Register the dataframe as a temp view for SQL operations
df_file1.createOrReplaceTempView("cclf_file1")

# COMMAND ----------

# Transform and load claim header data
print("Transforming claim header data")

try:
    # Using Spark SQL for transformation
    claim_header_sql = """
    SELECT
      CLM_ID AS claim_id,
      'A' AS claim_type,
      BENE_ID AS patient_id,
      PRVDR_NUM AS provider_id,
      ATT_PHYSN_NPI AS attending_provider_npi,
      OP_PHYSN_NPI AS operating_provider_npi,
      OTH_PHYSN_NPI AS other_provider_npi,
      to_date(CLM_FROM_DT, 'yyyyMMdd') AS service_start_date,
      to_date(CLM_THRU_DT, 'yyyyMMdd') AS service_end_date,
      to_date(CLM_ADMSN_DT, 'yyyyMMdd') AS admission_date,
      to_date(NCH_BENE_DSCHRG_DT, 'yyyyMMdd') AS discharge_date,
      CLM_IP_ADMSN_TYPE_CD AS admission_type_code,
      NCH_PTNT_STATUS_IND_CD AS patient_status_code,
      CLM_DISP_CD AS disposition_code,
      cast(CLM_PMT_AMT AS DECIMAL(15,2)) AS paid_amount,
      cast(CLM_TOT_CHRG_AMT AS DECIMAL(15,2)) AS total_charge_amount,
      cast(NCH_BENE_PTB_DDCTBL_AMT AS DECIMAL(15,2)) AS deductible_amount,
      cast(NCH_BENE_PTB_COINSRNC_AMT AS DECIMAL(15,2)) AS coinsurance_amount,
      cast(CLM_UTLZTN_DAY_CNT AS INT) AS utilization_day_count,
      CLM_SRVC_CLSFCTN_TYPE_CD AS service_classification_code,
      NCH_CLM_TYPE_CD AS bill_type_code,
      'CCLF_1' AS source_file_id,
      to_date('{run_date}', 'yyyy-MM-dd') AS source_file_date,
      current_timestamp() AS load_date
    FROM cclf_file1
    """.format(run_date=run_date)
    
    claim_header_df = spark.sql(claim_header_sql)
    
    # Write to staging table
    claim_header_df.write \
        .format("delta") \
        .mode("append") \
        .option("mergeSchema", "true") \
        .save(output_path + "stg_claim_header")
    
    print(f"Successfully loaded {claim_header_df.count()} records into stg_claim_header")
    
except Exception as e:
    print(f"Error processing claim header data: {str(e)}")
    raise

# COMMAND ----------

# Transform and load claim diagnosis data
print("Transforming claim diagnosis data")

try:
    # First, handle principal diagnosis
    principal_diag_sql = """
    SELECT
      generate_uuid() AS claim_diagnosis_id,
      CLM_ID AS claim_id,
      PRNCPAL_DGNS_CD AS diagnosis_code,
      PRNCPAL_DGNS_VRSN_CD AS diagnosis_code_version,
      0 AS diagnosis_sequence,
      'PRINCIPAL' AS diagnosis_type,
      PRNCPAL_DGNS_POA_IND_SW AS present_on_admission_flag,
      'CCLF_1' AS source_file_id,
      current_timestamp() AS load_date
    FROM cclf_file1
    WHERE PRNCPAL_DGNS_CD IS NOT NULL
    """
    
    principal_diag_df = spark.sql(principal_diag_sql)
    
    # Create list of secondary diagnoses to process
    secondary_diag_queries = []
    for i in range(1, 26):  # CMS CCLF has up to 25 secondary diagnoses
        secondary_diag_queries.append(f"""
        SELECT
          generate_uuid() AS claim_diagnosis_id,
          CLM_ID AS claim_id,
          ICD_DGNS_CD{i} AS diagnosis_code,
          ICD_DGNS_VRSN_CD{i} AS diagnosis_code_version,
          {i} AS diagnosis_sequence,
          'SECONDARY' AS diagnosis_type,
          ICD_DGNS_POA_IND_SW{i} AS present_on_admission_flag,
          'CCLF_1' AS source_file_id,
          current_timestamp() AS load_date
        FROM cclf_file1
        WHERE ICD_DGNS_CD{i} IS NOT NULL
        """)
    
    # Union all secondary diagnoses
    secondary_diag_sql = " UNION ALL ".join(secondary_diag_queries)
    secondary_diag_df = spark.sql(secondary_diag_sql)
    
    # Combine principal and secondary diagnoses
    all_diag_df = principal_diag_df.union(secondary_diag_df)
    
    # Write to staging table
    all_diag_df.write \
        .format("delta") \
        .mode("append") \
        .option("mergeSchema", "true") \
        .save(output_path + "stg_claim_diagnosis")
    
    print(f"Successfully loaded {all_diag_df.count()} records into stg_claim_diagnosis")
    
except Exception as e:
    print(f"Error processing claim diagnosis data: {str(e)}")
    raise

# COMMAND ----------

# Transform and load claim procedures data
print("Transforming claim procedures data")

try:
    # Create list of procedures to process
    proc_queries = []
    for i in range(1, 26):  # CMS CCLF has up to 25 procedures
        proc_queries.append(f"""
        SELECT
          generate_uuid() AS claim_procedure_id,
          CLM_ID AS claim_id,
          ICD_PRCDR_CD{i} AS procedure_code,
          ICD_PRCDR_VRSN_CD{i} AS procedure_code_version,
          {i} AS procedure_sequence,
          to_date(PRCDR_DT{i}, 'yyyyMMdd') AS procedure_date,
          'CCLF_1' AS source_file_id,
          current_timestamp() AS load_date
        FROM cclf_file1
        WHERE ICD_PRCDR_CD{i} IS NOT NULL
        """)
    
    # Union all procedures
    proc_sql = " UNION ALL ".join(proc_queries)
    proc_df = spark.sql(proc_sql)
    
    # Write to staging table
    proc_df.write \
        .format("delta") \
        .mode("append") \
        .option("mergeSchema", "true") \
        .save(output_path + "stg_claim_procedures")
    
    print(f"Successfully loaded {proc_df.count()} records into stg_claim_procedures")
    
except Exception as e:
    print(f"Error processing claim procedures data: {str(e)}")
    raise

# COMMAND ----------

# Extract revenue center details for claim lines
print("Transforming claim lines data")

try:
    claim_lines_sql = """
    WITH revenue_centers AS (
      SELECT
        CLM_ID,
        REV_CNTR,
        REV_CNTR_CD,
        HCPCS_CD,
        HCPCS_1ST_MDFR_CD,
        HCPCS_2ND_MDFR_CD,
        HCPCS_3RD_MDFR_CD,
        HCPCS_4TH_MDFR_CD,
        REV_CNTR_UNIT_CNT,
        REV_CNTR_RATE_AMT,
        REV_CNTR_PMT_AMT_AMT,
        REV_CNTR_TOT_CHRG_AMT,
        REV_CNTR_NCVRD_CHRG_AMT,
        REV_CNTR_DDCTBL_COINSRNC_CD,
        REV_CNTR_NDC_QTY,
        REV_CNTR_NDC_QTY_QLFR_CD,
        from_unixtime(unix_timestamp(CLM_FROM_DT, 'yyyyMMdd')) AS service_date
      FROM cclf_file1
      LATERAL VIEW explode(
        arrays_zip(
          REV_CNTR_CD,
          HCPCS_CD,
          HCPCS_1ST_MDFR_CD,
          HCPCS_2ND_MDFR_CD,
          HCPCS_3RD_MDFR_CD,
          HCPCS_4TH_MDFR_CD,
          REV_CNTR_UNIT_CNT,
          REV_CNTR_RATE_AMT,
          REV_CNTR_PMT_AMT_AMT,
          REV_CNTR_TOT_CHRG_AMT,
          REV_CNTR_NCVRD_CHRG_AMT,
          REV_CNTR_DDCTBL_COINSRNC_CD,
          REV_CNTR_NDC_QTY,
          REV_CNTR_NDC_QTY_QLFR_CD
        )
      ) AS rev_cntr
    )
    
    SELECT
      generate_uuid() AS claim_line_id,
      CLM_ID AS claim_id,
      ROW_NUMBER() OVER (PARTITION BY CLM_ID ORDER BY REV_CNTR_CD) AS line_number,
      HCPCS_CD AS hcpcs_code,
      HCPCS_1ST_MDFR_CD AS hcpcs_modifier_1,
      HCPCS_2ND_MDFR_CD AS hcpcs_modifier_2,
      HCPCS_3RD_MDFR_CD AS hcpcs_modifier_3,
      HCPCS_4TH_MDFR_CD AS hcpcs_modifier_4,
      REV_CNTR_CD AS revenue_code,
      cast(REV_CNTR_PMT_AMT_AMT AS DECIMAL(15,2)) AS line_payment_amount,
      cast(REV_CNTR_RATE_AMT AS DECIMAL(15,2)) AS line_allowed_amount,
      NULL AS line_coinsurance_amount,
      NULL AS line_deductible_amount,
      NULL AS line_primary_payer_paid_amount,
      cast(REV_CNTR_UNIT_CNT AS DECIMAL(8,2)) AS line_service_units,
      service_date AS line_service_date,
      NULL AS line_place_of_service_code,
      NULL AS line_service_type_code,
      NULL AS line_processing_indicator,
      NULL AS line_national_drug_code,
      'CCLF_1' AS source_file_id,
      current_timestamp() AS load_date
    FROM revenue_centers
    """
    
    claim_lines_df = spark.sql(claim_lines_sql)
    
    # Write to staging table
    claim_lines_df.write \
        .format("delta") \
        .mode("append") \
        .option("mergeSchema", "true") \
        .save(output_path + "stg_claim_lines")
    
    print(f"Successfully loaded {claim_lines_df.count()} records into stg_claim_lines")
    
except Exception as e:
    print(f"Error processing claim lines data: {str(e)}")
    raise

# COMMAND ----------

print("CCLF File 1 transformation completed successfully")