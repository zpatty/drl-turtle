#include <Wire.h>
#include <Adafruit_INA219.h>
#include "ICM_20948.h" // Click here to get the library: http://librarymanager/All#SparkFun_ICM_20948_IMU
#include "MS5837.h"

#define AD0_VAL 1

Adafruit_INA219 ina219;
ICM_20948_I2C icm; 
icm_20948_DMP_data_t data;
double q1 = 0;
double q2 = 0;
double q3 = 0;
double q0 = 0;

double base_time = 0;
double now_time = 0;
double last_time = 0;
double loop_time = 0;

double depth_val = 0;

MS5837 ms_depth;

void setup(void) 

{

  Serial.begin(115200);

  while (!Serial) {

      delay(1);

  }
  Serial.println("Serial Begin");
  Wire.begin();
  Wire.setClock(100000);
  Serial.println("Wire init");
  icm.begin(Wire, AD0_VAL);
  Serial.println("ICM Begin");
  if (!ina219.begin() || icm.status != ICM_20948_Stat_Ok) {

    Serial.println(F("Failed to find ICM chip"));

    while (1) { delay(10); }
  }
  Serial.println("ICM Initialized");
  while (!ms_depth.init()) {
    Serial.println("Depth init failed!");
    Serial.println("Are SDA/SCL connected correctly?");
    Serial.println("Blue Robotics Bar30: White=SDA, Green=SCL");
    Serial.println("\n\n\n");
    delay(5000);
  }
  Serial.println("Depth Initialized");
  ms_depth.setModel(MS5837::MS5837_30BA);
  ms_depth.setFluidDensity(997); // kg/m^3 (freshwater, 1029 for seawater)

  bool success = true;
    // Initialize the DMP. initializeDMP is a weak function. You can overwrite it if you want to e.g. to change the sample rate
  success &= (icm.initializeDMP() == ICM_20948_Stat_Ok);

  // Enable the DMP orientation sensor, gyroscope, and accelerometer
  success &= (icm.enableDMPSensor(INV_ICM20948_SENSOR_ORIENTATION) == ICM_20948_Stat_Ok);

  // E.g. For a 5Hz ODR rate when DMP is running at 55Hz, value = (55/5) - 1 = 10.
  success &= (icm.setDMPODRrate(DMP_ODR_Reg_Quat9, 0) == ICM_20948_Stat_Ok); // Set to the maximum

  // Enable the FIFO
  success &= (icm.enableFIFO() == ICM_20948_Stat_Ok);

  // Enable the DMP
  success &= (icm.enableDMP() == ICM_20948_Stat_Ok);

  // Reset DMP
  success &= (icm.resetDMP() == ICM_20948_Stat_Ok);

  // Reset FIFO
  success &= (icm.resetFIFO() == ICM_20948_Stat_Ok);

  if (success){

    Serial.println(F("DMP enabled!"));

  }else{
    Serial.println(F("Enable DMP Failed!"));
    // Serial.println(F("Please make sure you uncommented line 29 (#define ICM_20948_USE_DMP) in ICM_20948_C.h..."));
    while(1){
      ; // Do nothing more 
    }

  }

  base_time = millis();
}


void loop(void) 

{
  printIMUData(&icm, &ina219);
  // ms_depth.read();
  
  // Serial.print("Depth: "); 
  // Serial.print(ms_depth.depth()); 
  // Serial.println(" m");
  
}

void printScaledAGMT(ICM_20948_I2C *sensor) {
  //Scaled. Acc (mg)
  Serial.print("\"Acc\":[ ");
  // printFormattedFloat(sensor->accX(), 5, 2);
  Serial.print(sensor->accX());

  Serial.print(", ");
  // printFormattedFloat(sensor->accY(), 5, 2);
  Serial.print(sensor->accY());

  Serial.print(", ");
  // printFormattedFloat(sensor->accZ(), 5, 2);
  Serial.print(sensor->accZ());

  // Gyr (DPS)
  Serial.print(" ], \"Gyr\" :[ ");
  printFormattedFloat(sensor->gyrX(), 5, 2);
  Serial.print(", ");
  printFormattedFloat(sensor->gyrY(), 5, 2);
  Serial.print(", ");
  printFormattedFloat(sensor->gyrZ(), 5, 2);
  Serial.print(" ],");
  // Serial.println();
}

void printFormattedFloat(float val, uint8_t leading, uint8_t decimals) {
  float aval = abs(val);

  if (val < 0){
    Serial.print("-");
  }
  else{
    Serial.print(" ");
  }

  for (uint8_t indi = 0; indi < leading; indi++){
    uint32_t tenpow = 0;
    if (indi < (leading - 1)){
      tenpow = 1;
    }
    for (uint8_t c = 0; c < (leading - 1 - indi); c++){
      tenpow *= 10;
    }
    // if (aval < tenpow){
    //   Serial.print("0");
    // }
    // else{
    //   break;
    // }
  }

  if (val < 0){
    Serial.print(-val, decimals);
  }
  else{
    Serial.print(val, decimals);
  }
  
}


void printIMUData(ICM_20948_I2C *sensor, Adafruit_INA219 * volt_sensor){
  icm.readDMPdataFromFIFO(&data);
  
  now_time = millis();
  loop_time = now_time - last_time;
  last_time = now_time;
  ms_depth.read();
  depth_val = ms_depth.depth();
  // if (now_time - base_time > 50) {
  //   ms_depth.read();
  //   // Serial.print("Depth: "); 
  //   depth_val = ms_depth.depth();
  //   // Serial.print(ms_depth.depth()); 
  //   // Serial.println(" m");
  //   base_time = millis();
  // }
  // else {
  //   depth_val = NAN;
  // }
  
  float busvoltage = 0;

  if ((icm.status == ICM_20948_Stat_Ok) || (icm.status == ICM_20948_Stat_FIFOMoreDataAvail)) // Was valid data available?
  {

    if ((data.header & DMP_header_bitmap_Quat9) > 0) // We have asked for orientation data so we should receive Quat9
    {
      // Q0 value is computed from this equation: Q0^2 + Q1^2 + Q2^2 + Q3^2 = 1.
      // In case of drift, the sum will not add to 1, therefore, quaternion data need to be corrected with right bias values.
      // The quaternion data is scaled by 2^30.  if (myICM.status != ICM_20948_Stat_FIFOMoreDataAvail) // If more data is available then we should read it right away - and not delay

      q1 = ((double)data.Quat9.Data.Q1) / 1073741824.0; // Convert to double. Divide by 2^30
      q2 = ((double)data.Quat9.Data.Q2) / 1073741824.0; // Convert to double. Divide by 2^30
      q3 = ((double)data.Quat9.Data.Q3) / 1073741824.0; // Convert to double. Divide by 2^30
      q0 = sqrt(1.0 - ((q1 * q1) + (q2 * q2) + (q3 * q3)));
      Serial.print(" { \"Quat\": [ ");
      Serial.print(q0, 3);
      Serial.print(", ");
      Serial.print(q1, 3);
      Serial.print(", ");
      Serial.print(q2, 3);
      Serial.print(", ");
      Serial.print(q3, 3);
      Serial.print(" ],");
      sensor->getAGMT();
      Serial.print("\"Acc\":[ ");
      // printFormattedFloat(sensor->accX(), 5, 2);
      Serial.print(sensor->accX()/1000);

      Serial.print(", ");
      // printFormattedFloat(sensor->accY(), 5, 2);
      Serial.print(sensor->accY()/1000);

      Serial.print(", ");
      // printFormattedFloat(sensor->accZ(), 5, 2);
      Serial.print(sensor->accZ()/1000);
      // Gyr (DPS)
      Serial.print(" ], \"Gyr\" :[ ");
      printFormattedFloat(sensor->gyrX(), 5, 2);
      Serial.print(", ");
      printFormattedFloat(sensor->gyrY(), 5, 2);
      Serial.print(", ");
      printFormattedFloat(sensor->gyrZ(), 5, 2);

      // Depth
      Serial.print(" ], \"Depth\" :[ ");
      printFormattedFloat(depth_val, 5, 2);
      Serial.print(" ],");

      busvoltage = volt_sensor->getBusVoltage_V();
      Serial.print("\"Voltage\": [ ");
      Serial.print(busvoltage);

      // // Time
      // Serial.print(" ], \"Time\" :[ ");
      // printFormattedFloat(loop_time, 5, 2);
      Serial.print(" ]}");
      Serial.println("");

    }
  }
  else{
    Serial.println("no data");
  }
  delay(3);

} 
