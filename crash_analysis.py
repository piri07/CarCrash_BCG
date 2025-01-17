import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, rank, desc
from pyspark.sql.window import Window

class CarCrashAnalytics:
    """
    Class to handle car crash analytics using Spark DataFrame APIs.

    Attributes:
        spark (SparkSession): Spark session for data processing.
        datasets (dict): Dictionary to hold loaded datasets.
    """

    def __init__(self, data_folder):
        """Initilize spark session"""
        self.spark = SparkSession.builder \
            .master("local[*]") \
            .appName("CarCrashAnalytics") \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        self.data_folder = data_folder
        self.datasets = {}

    def load_datasets(self):
        """Dynamically load all CSV files from the specified folder into memory."""
        for file_path in os.listdir(self.data_folder):
            if file_path.endswith(".csv"):
                filename = os.path.splitext(file_path)[0]
                self.datasets[filename] = self.spark.read.csv(os.path.join(self.data_folder, file_path), header=True, inferSchema=True)

    def analytics_1(self):
        """Find the number of crashes where males killed > 2."""
        df = self.datasets.get('Primary_Person_use')
        filtered_df = df.filter((col("PRSN_GNDR_ID") == "MALE") & (col("PRSN_INJRY_SEV_ID") == "KILLED"))
        grouped_df = filtered_df.groupBy("CRASH_ID").count()
        result_df = grouped_df.filter(col("count") >= 2)
        crash_count = result_df.count()
        print(f"Number of crashes where males killed >= 2: {crash_count}")

    def analytics_2(self):
        """How many two-wheelers are booked for crashes?"""
        df = self.datasets.get('Units_use')
        filtered_df = df.filter((col("VEH_BODY_STYL_ID") == "MOTORCYCLE") | (col("VEH_BODY_STYL_ID") == "POLICE MOTORCYCLE"))
        unique_df = filtered_df.dropDuplicates(['CRASH_ID'])
        two_wheelers_crashed = unique_df.count()
        print(f"Number of crashes involving 2 wheelers: {two_wheelers_crashed}")

    def analytics_3(self):
        """Top 5 Vehicle Makes (driver died, airbags not deployed)."""
        df1 = self.datasets.get('Primary_Person_use')
        df2 = self.datasets.get('Units_use')
        df_joined = df1.join(df2, on="CRASH_ID", how="inner")
        df_joined = df_joined.filter((col("PRSN_TYPE_ID") == "DRIVER") & (col("PRSN_AIRBAG_ID") == "NOT DEPLOYED") & (col("PRSN_INJRY_SEV_ID") == "KILLED"))
        grouped_df = df_joined.groupBy("VEH_MAKE_ID").count()
        top_5_df = grouped_df.orderBy(col("count").desc()).limit(5)
        top_5_values = [row['VEH_MAKE_ID'] for row in top_5_df.collect()]
        print("Top 5 Vehicle Makes where airbags did not deploy and driver died are:")
        print(top_5_values)

    def analytics_4(self):
        """Vehicles with valid license involved in hit and run."""
        df1 = self.datasets.get('Primary_Person_use')
        df2 = self.datasets.get('Charges_use')

        df_joined = df1.join(df2, on="CRASH_ID", how="inner")
        filter_df = df_joined.filter(
            (col("DRVR_LIC_CLS_ID") != "UNLICENSED") &
            (col("DRVR_LIC_CLS_ID") != "UNKNOWN") &
            (col("PRSN_TYPE_ID") == "DRIVER")
        )

        hit_and_run_df = filter_df.filter(col("CHARGE").contains("HIT AND RUN"))

        hit_and_run_df = hit_and_run_df.dropDuplicates(['CRASH_ID'])

        # Count the number of distinct crashes involving vehicles with valid licenses and hit and run charges
        no_of_vehicles = hit_and_run_df.count()
        print(f"Number of vehicles involved in a crash with valid driver's license and hit and run: {no_of_vehicles}")


    def analytics_5(self):
        """State with the highest non-female accidents."""
        df = self.datasets.get('Primary_Person_use')
        df_filtered = df.filter((col("PRSN_GNDR_ID") != "FEMALE"))
        accidents_count_df = df_filtered.groupBy("DRVR_LIC_STATE_ID").agg(count("CRASH_ID").alias("accident_count"))
        max_accident_state = accidents_count_df.orderBy(col("accident_count").desc()).limit(1)
        max_accident_state_value = max_accident_state.collect()[0]["DRVR_LIC_STATE_ID"]
        max_accident_state_count = max_accident_state.collect()[0]["accident_count"]
        print(f"State with highest non-female accident is: {max_accident_state_value}")
        print(f"Number of crashes: {max_accident_state_count}")

    def analytics_6(self):
        """Top 3rd to 5th Vehicle Makes causing the largest number of injuries."""
        df1 = self.datasets.get('Primary_Person_use')
        df2 = self.datasets.get('Units_use')
        df_joined = df1.join(df2, on="CRASH_ID", how="inner")
        df_joined = df_joined.filter((col("PRSN_INJRY_SEV_ID") != "NOT INJURED") & (col("PRSN_INJRY_SEV_ID") != "UNKNOWN") & (col("PRSN_INJRY_SEV_ID") != "NA"))
        unique_df = df_joined.dropDuplicates(["CRASH_ID"])
        injury_count_df = unique_df.groupBy("VEH_MAKE_ID").agg(count("CRASH_ID").alias("injury_count"))
        sorted_df = injury_count_df.orderBy(col("injury_count").desc())
        top_3_to_5_df = sorted_df.limit(5).subtract(sorted_df.limit(2))
        top_3_to_5 = top_3_to_5_df.collect()
        print("Top 3rd to 5th Vehicle Makes causing the largest number of injuries including death:")
        for row in top_3_to_5:
            print(f"Vehicle Make: {row['VEH_MAKE_ID']}, Number of Injuries/Deaths: {row['injury_count']}")

    def analytics_7(self):
        """Top ethnic groups for each vehicle body type."""
        df1 = self.datasets.get('Primary_Person_use')
        df2 = self.datasets.get('Units_use')
        df_joined = df1.join(df2, on="CRASH_ID", how="inner")
        df_joined = df_joined.filter((col("VEH_BODY_STYL_ID") != "UNKNOWN") & (col("VEH_BODY_STYL_ID") != "NA") & (col("VEH_BODY_STYL_ID") != "NOT REPORTED"))
        ethnicity_count_df = df_joined.groupBy("VEH_BODY_STYL_ID", "PRSN_ETHNICITY_ID").agg(count("CRASH_ID").alias("ethnicity_count"))
        top_ethnic_group_df = ethnicity_count_df.withColumn("rank", rank().over(Window.partitionBy("VEH_BODY_STYL_ID").orderBy(desc("ethnicity_count"))))
        top_ethnic_group_df = top_ethnic_group_df.filter(col("rank") == 1).drop("rank")
        top_ethnic_group = top_ethnic_group_df.collect()
        print("Top ethnic groups for each vehicle body type:")
        for row in top_ethnic_group:
            print(f"Body Style: {row['VEH_BODY_STYL_ID']}, Top Ethnic Group: {row['PRSN_ETHNICITY_ID']}, Count: {row['ethnicity_count']}")

    def analytics_8(self):
        """Top 5 ZIP codes with alcohol as the contributing factor to a crash."""
        df = self.datasets.get('Primary_Person_use')
        df_filtered = df.filter((col("PRSN_ALC_RSLT_ID") == "Positive") & (col("DRVR_ZIP") != "NULL"))
        unique_df = df_filtered.dropDuplicates(["CRASH_ID"])
        top_5_zip_codes = unique_df.groupby("DRVR_ZIP").count().orderBy(desc("count")).limit(5)
        top_5_zip_codes_list = top_5_zip_codes.collect()
        print("Top 5 ZIP codes with alcohol as the contributing factor to a crash:")
        for row in top_5_zip_codes_list:
            print(f"Driver Zip Code: {row['DRVR_ZIP']}, Count: {row['count']}")

    def analytics_9(self):
      df1 = self.datasets.get('Primary_Units_use')
      df2 = self.datasets.get('Damages_use')
      allowed_rating = ["DAMAGED 5","DAMAGED 6","DAMAGED 7 HIGHEST"]
      insured  =["PROOF OF LIABILITY INSURANCE","LIABILITY INSURANCE POLICY","SURETY BOND","CERTIFICATE OF SELF-INSURANCE"]
      df_joined = df1.join(df2, on="CRASH_ID", how="inner")
      #taking ratings of both veh_dmag_scl_1 and veh_dmag_scl_2 results in None dataframe
      df_joined = df_joined.filter((col("DAMAGED_PROPERTY").contains("NO DAMAGE"))& (col("VEH_DMAG_SCL_1_ID").isin(allowed_rating))& (col("FIN_RESP_TYPE_ID").isin(insured)))#& (col("VEH_DMAG_SCL_2_ID").isin(allowed_rating)))
      unique_df = df_joined.dropDuplicates(["CRASH_ID"])
      count = unique_df.count()
      print(f"Distinct Crash IDs where there was no damage and the car availed insurance are: {count}")

    def analytics_10(self):
      df1 = self.datasets.get("df_Units_use")
      df2 = self.datasets.get("df_Charges_use")
      df3 = self.datasets.get("df_Primary_Person_use")
      df_driver_offence = df3.join(df2, on="CRASH_ID", how="inner")
      df_driver_offence = df_driver_offence.filter((col("PRSN_TYPE_ID")=="DRIVER")& (col("CHARGE").contains("SPEED")))

      #finding the top 25 states with offences
      states_with_offence = df3.dropDuplicates(['CRASH_ID'])
      states_with_offence = states_with_offence.groupby("DRVR_LIC_STATE_ID").agg(count("CRASH_ID").alias("crash_count"))
      states_with_offence = states_with_offence.filter((col("DRVR_LIC_STATE_ID")!="NA")&(col("DRVR_LIC_STATE_ID")!="Unknown")&(col("DRVR_LIC_STATE_ID")!="Other"))
      states_with_offence = states_with_offence.orderBy(desc("crash_count")).limit(25)

      #filtering out the crashes happening in top 25 states
      top_25_states_list = [row["DRVR_LIC_STATE_ID"] for row in states_with_offence.collect()]
      df_driver_offence = df_driver_offence.filter(col("DRVR_LIC_STATE_ID").isin(top_25_states_list))

      #filtering the top 10 colours used by cars
      Top_10_vehicle_colors = df1.dropDuplicates(['CRASH_ID'])
      Top_10_vehicle_colors = Top_10_vehicle_colors.groupby("VEH_COLOR_ID").agg(count("CRASH_ID").alias("crash_count"))
      Top_10_vehicle_colors = Top_10_vehicle_colors.orderBy(desc("crash_count")).limit(10)

      #filtering the crashes using with vehicles having these colours.
      Top_10_vehicle_colors_list = [row["VEH_COLOR_ID"] for row in Top_10_vehicle_colors.collect()]
      df1 = df1.filter(col("VEH_COLOR_ID").isin(Top_10_vehicle_colors_list))

      #joing the above two dataframes and gettign the top 5 vehicle makers
      final_df = df1.join(df_driver_offence, on="CRASH_ID", how="inner")
      final_df = final_df.dropDuplicates(['CRASH_ID'])
      Top_5_vehicles = final_df.groupby("VEH_MAKE_ID").agg(count("CRASH_ID").alias("Vehicle_count"))
      Top_5_vehicles = Top_5_vehicles.orderBy(desc("Vehicle_count")).limit(5)
      top_5_vehicles_list = [row["VEH_MAKE_ID"] for row in Top_5_vehicles.collect()]
      print("Top 5 Vehicle Makes where drivers are charged with speeding related offences, has licensed Drivers, used top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of offences are:")
      for i in top_5_vehicles_list:
        print(i)


def menu():
    """Command-line menu to execute car crash analytics."""
    data_folder = "./Data/"  # Folder containing the datasets
    analytics = CarCrashAnalytics(data_folder)
    analytics.load_datasets()
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("====================== Car Crash Analytics =====================")
        print("1. Find crashes where males killed > 2")
        print("2. How many two-wheelers are booked for crashes?")
        print("3. Determine Top 5 Vehicle Makes (driver died, airbags not deployed)")
        print("4. Vehicles with valid license involved in hit and run")
        print("5. State with the highest non-female accidents")
        print("6. Top 3rd to 5th Vehicle Makes causing injuries/deaths")
        print("7. Top ethnic groups for each vehicle body type")
        print("8. Top 5 ZIP codes with alcohol as contributing factor")
        print("9. Count of Distinct Crash IDs where No Damaged Property was observed and Damage Levelis above 4 and car avails Insurance")
        print("10. Top 5 Vehicle Makes where drivers are charged with speeding related offences, has licensed Drivers, used top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of offences")

        choice = input("Enter your choice: ")
        if choice == "1":
            analytics.analytics_1()
        elif choice == "2":
            analytics.analytics_2()
        elif choice == "3":
            analytics.analytics_3()
        elif choice == "4":
            analytics.analytics_4()
        elif choice == "5":
            analytics.analytics_5()
        elif choice == "6":
            analytics.analytics_6()
        elif choice == "7":
            analytics.analytics_7()
        elif choice == "8":
            analytics.analytics_8()
        elif choice == "9":
            analytics.analytics_9()
        elif choice == "10":
            analytics.analytics_10()
        elif choice == "11":
            print("Exiting")
            break
        else:
            print("Invalid choice. Please try again.")

        input("Press Enter to continue...")

if __name__ == "__main__":
    menu()

