/**
 * @file main.cpp
 * @author Cherntay Shih, Joseph Leandre Derpo, and Nonthakorn
 * @brief Weather station predicting weather conditions with limited data - TF2.1.1 to TFlite
 * @version 0.1
 * @date 2022-04-03
 * 
 * @copyright Copyright (c) 2022
 */

//Libraries
#include <DHT.h>
#include <WiFi.h>
#include <Arduino.h>
#include "esp_wifi.h"
#include <WiFiMulti.h>
#include <time.h>

#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

//Our model
#include "ANN_WeatherModel_Two_Params.h"

//##// -> Initialization <- //##//
#define WIFI_SSID ""
#define WIFI_PASSWORD ""
#define INFLUXDB_URL ""
#define INFLUXDB_TOKEN ""
#define INFLUXDB_ORG ""
#define INFLUXDB_BUCKET ""
#define TZ_INFO "ICT-7"   // Bangkok Timezone
#define DEVICE "ESP32"

#include <InfluxDbClient.h>
#include <InfluxDbCloud.h>

// InfluxDB client instance with preconfigured InfluxCloud certificate
WiFiMulti wifiMulti;
InfluxDBClient client(INFLUXDB_URL, INFLUXDB_ORG, INFLUXDB_BUCKET, INFLUXDB_TOKEN, InfluxDbCloud2CACert);
Point sensor("weather_sensor"); // Data point

// Figure out whats going in our model
#define DEBUG 0

//Constants
#define DHTPIN 32   // what pin we're connected to 50
#define DHTTYPE DHT22   // DHT 22  (AM2302)
DHT dht(DHTPIN, DHTTYPE); // Initialize DHT sensor for normal 16mhz Arduino

#define uS_TO_S_FACTOR 1000000 //Conversion factor for micro second to second
#define TIME_TO_SLEEP 60 // Time ESP32 will go to sleep in seconds

// Variables
float hum;  //Stores humidity value
float temp; //Stores temperature value
float heatIndex;

int num_loops;
int connection_time_counter;
int THRESHOLD_CONNECTION_TIME = 100;

// Amount of labels i.e. 4
constexpr int LABEL_COUNTS = 4;

// Temp variables
float output_score;
float output_max_score;
byte index_numbers; 

// TF Model Outputs Labels
const char* LABELS[LABEL_COUNTS] = {
  "cloudy", //0
  "foggy", //1
  "rainy", //2
  "sunny" //3
};

// TFLite Globals, used for compatibility with Arduino-style sketches.
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  // tflite::AllOpsResolver tflOpsResolver;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  // Create an area of memory to use for input,output, and other TensorFlow
  // arrays. It will need to be adjusted by compiling, running, and looking
  // for errors.
  constexpr int kTensorArenaSize = 12 * 1024; //constexpr int kTensorArenaSize = 1024*25;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

// Read sensor data
void sensor_reading(){
  temp = dht.readTemperature(false);
  hum = dht.readHumidity();
  heatIndex = dht.computeHeatIndex(temp, hum, false);
}

// Debug TFlite
void tflite_debug(){
  Serial.print("Number of Dimensions: ");
  Serial.println(model_input->dims->size);

  Serial.print("Input Dim 1 size: ");
  Serial.println(model_input->dims->data[0]);

  Serial.print("Input Dim 2 size: ");
  Serial.println(model_input->dims->data[1]);

  Serial.print("Input Input type: ");
  Serial.println(model_input->type);

  Serial.println("Number of Output Dimensions: ");
  Serial.println(model_output->dims->size);

  Serial.print("Output Dim 1 size: ");
  Serial.println(model_output->dims->data[0]);

  Serial.print("Output Dim 2 size: ");
  Serial.println(model_output->dims->data[1]);

  Serial.print("Output type: ");
  Serial.println(model_output->type);
}

//Print serialprint easier way
void StreamPrint_progmem(Print &out,PGM_P format,...)
{
  // program memory version of printf - copy of format string and result share a buffer
  // so as to avoid too much memory use
  char formatString[128], *ptr;
  strncpy_P( formatString, format, sizeof(formatString) ); // copy in from program mem
  // null terminate - leave last char since we might need it in worst case for result's \0
  formatString[ sizeof(formatString)-2 ]='\0'; 
  ptr=&formatString[ strlen(formatString)+1 ]; // our result buffer...
  va_list args;
  va_start (args,format);
  vsnprintf(ptr, sizeof(formatString)-1-strlen(formatString), formatString, args );
  va_end (args);
  formatString[ sizeof(formatString)-1 ]='\0'; 
  out.print(ptr);
}
#define Serialprint(format, ...) StreamPrint_progmem(Serial,PSTR(format),##__VA_ARGS__)

// Debug output score of the model
void debug_output_score()
{
  Serialprint("Output score [%d]: %f\n", num_loops, output_score);
}

void setup_WiFi()
{
    //# Setup WiFi and sync time #//
    WiFi.mode(WIFI_STA);
    wifiMulti.addAP(WIFI_SSID, WIFI_PASSWORD);
    configTime(7*60*60, 0, "pool.ntp.org");   // GMT+7 Time , No DST

    Serialprint("Connecting to wifi");
    
    while (wifiMulti.run() != WL_CONNECTED) {
      Serialprint(".");
      delay(100);
       connection_time_counter += connection_time_counter;
       if(connection_time_counter >= THRESHOLD_CONNECTION_TIME){
         ESP.restart();
      }
    }
    Serial.println();

    sensor.addTag("device", DEVICE);  // Add device tags
    //# Sync Time #//
    timeSync(TZ_INFO, "pool.ntp.org", "time.nis.gov");  // Sync device time
}

void check_Influx(){
    //# Check InfluxDB connection #//
    if (client.validateConnection()) {
      Serialprint("Connected to InfluxDB: ");
      Serial.println(client.getServerUrl());
    }
    else {
      Serialprint("InfluxDB connection failed: ");
      Serial.println(client.getLastErrorMessage());
    }
}

void publish_data(){
  
    //# Send data to InfluxDB #//
    sensor.clearFields();
    
    sensor.addField("Temperature (C)", temp);
    sensor.addField("Humidity", hum);
    sensor.addField("Weather Condition", LABELS[index_numbers]);
    sensor.addField("Heat Index", heatIndex);
    sensor.addField("Index Numbers", index_numbers);
    
    // Print data that we sent on serial monitor
    Serial.print("Writing: ");
    Serial.println(client.pointToLineProtocol(sensor));

}

void check_WiFi()
{
    //# Check WiFi connection and reconnect if needed #//
    if (wifiMulti.run() != WL_CONNECTED) {
      Serialprint("Wifi connection lost\n");
    }
    //# Check if it can't send data to InfluxDB #//
    if (!client.writePoint(sensor)) {
      Serialprint("InfluxDB write failed: ");
      Serial.println(client.getLastErrorMessage());
    }
}



void setup() {
  Serial.begin(115200);

  // Have DHT sensor begin 
  dht.begin();
  
  // wait for Serial to connect
#if DEBUG 
  while(!Serial);
#endif

  // Set up logging (will report to Serial, even within TfLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(ANN_WeatherModel_Two_Params_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
      "Model provided is schema version %d not equal "
      "to supported version %d.",
      model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Input all necessary operators for TFlite ESP32
  static tflite::ops::micro::AllOpsResolver resolver;

  // This pulls in all the operation implementations we need.
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize,error_reporter);
  interpreter = &static_interpreter;


  // Allocate memory from the tensor arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Assign model input and output bufers (tensors) to pointers
  model_input = interpreter ->input(0);
  model_output = interpreter->output(0);

  #if DEBUG
    tflite_debug();
  #endif

  setup_WiFi();
  check_Influx();
  delay(3000);
}


void loop() {
  delay(500);

  // Parsing the model input to buffer
  model_input->data.f[0] = temp;
  model_input->data.f[1] = hum;

  // Run inference and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Parsing the model output from buffer
  for (byte i = 0; i < LABEL_COUNTS; i++)
  {
    output_score = model_output->data.f[i];
    num_loops = i + 1;
    #if DEBUG
    debug_output_score();
    #endif

    if ((i == 0) || (output_score > output_max_score))
    {
      output_max_score =  output_score;
      index_numbers = i;
    }
  }
  
#if DEBUG
   TF_LITE_REPORT_ERROR(error_reporter, "Weather: %s, [%d]", LABELS[index_numbers], output_max_score);
#endif
  sensor_reading();
  Serial.print("Humidity: ");
  Serial.print(hum);
  Serial.print(" %, Temp: ");
  Serial.print(temp);
  Serial.println(" Celsius");
  Serialprint("Weather Condition: %c ", LABELS[index_numbers]);
  Serialprint("Index Numbers: %d", index_numbers);
  
  // Reset the max score
  output_max_score = 0;
  
  delay(500);

  // Publish the data and prediction to influxdb then visualize on Grafana
  publish_data();
  check_WiFi();
  delay(500);

  Serial.print("going to sleep!");
  delay(500);
  esp_sleep_enable_timer_wakeup(TIME_TO_SLEEP * uS_TO_S_FACTOR);
  esp_deep_sleep_start();
}