#include <Wire.h>
#include <Adafruit_INA219.h>
#include "ICM_20948.h" // Click here to get the library: http://librarymanager/All#SparkFun_ICM_20948_IMU

#define AD0_VAL 1

Adafruit_INA219 ina219;
ICM_20948_I2C icm; 

void setup(void) 

{

  Serial.begin(115200);

  while (!Serial) {

      // will pause Zero, Leonardo, etc until serial console opens

      delay(1);

  }
  Wire.begin();
  Wire.setClock(400000);
  // Initialize the INA219.
  // By default the initialization will use the largest range (32V, 2A).  However
  // you can call a setCalibration function to change this range (see comments).
  icm.begin(Wire, AD0_VAL);
  if (!ina219.begin() || icm.status != ICM_20948_Stat_Ok) {

    Serial.println("Failed to find INA219 chip");

    while (1) { delay(10); }

  }

  // To use a slightly lower 32V, 1A range (higher precision on amps):

  //ina219.setCalibration_32V_1A();

  // Or to use a lower 16V, 400mA range (higher precision on volts and amps):

  //ina219.setCalibration_16V_400mA();



}



void loop(void) 

{

  // float shuntvoltage = 0;
  float busvoltage = 0;
  // float current_mA = 0;
  // float loadvoltage = 0;
  // float power_mW = 0;

  // shuntvoltage = ina219.getShuntVoltage_mV();
  busvoltage = ina219.getBusVoltage_V();
  // current_mA = ina219.getCurrent_mA();
  // power_mW = ina219.getPower_mW();
  // loadvoltage = busvoltage + (shuntvoltage / 1000);

  icm.getAGMT();         // The values are only updated when you call 'getAGMT'
  printScaledAGMT(&icm); // This function takes into account the scale settings from when the measurement was made to calculate the values with units
  // Bus Voltage (V)
  Serial.print("\"Voltage\": [ ");
  Serial.print(busvoltage);
  Serial.print(" ]}");
  Serial.println("");
  delay(10);

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

void printScaledAGMT(ICM_20948_I2C *sensor) {
  //Scaled. Acc (mg)
  Serial.print("{\"Acc\":[ ");
  printFormattedFloat(sensor->accX(), 5, 2);
  Serial.print(", ");
  printFormattedFloat(sensor->accY(), 5, 2);
  Serial.print(", ");
  printFormattedFloat(sensor->accZ(), 5, 2);
  // Gyr (DPS)
  Serial.print(" ], \"Gyr\" :[ ");
  printFormattedFloat(sensor->gyrX(), 5, 2);
  Serial.print(", ");
  printFormattedFloat(sensor->gyrY(), 5, 2);
  Serial.print(", ");
  printFormattedFloat(sensor->gyrZ(), 5, 2);
  // Mag (uT)
  Serial.print(" ], \"Mag\" :[ ");
  printFormattedFloat(sensor->magX(), 5, 2);
  Serial.print(", ");
  printFormattedFloat(sensor->magY(), 5, 2);
  Serial.print(", ");
  printFormattedFloat(sensor->magZ(), 5, 2);
  // Serial.print(" ], Tmp (C) [ ");
  // printFormattedFloat(sensor->temp(), 5, 2);
  Serial.print(" ],");
  // Serial.println();
}