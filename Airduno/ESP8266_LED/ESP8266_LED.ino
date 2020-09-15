#include "ESP8266.h"

#define ESP_8266_SSID        "test"
#define ESP_8266_PASSWORD    "12345678"
#define ESP8266_Reset_Pin 12
#define ESP_8266_HOST_PORT   8090
#define ESP8266_RX 10
#define ESP8266_TX 11
#define Max_Delay 5

#define ESP8266_Command_ON ",ONON!,"
#define ESP8266_Command_BLINK ",BKBK!,"
#define ESP8266_Command_OFF ",OFOF!,"

#define Max_Delay_Cnt_LED 10 
#define Max_Delay_Cnt_ESP8266 500

// LED
char LED_PIN = 13;
int State=0, Prev_State=0;    // 0: off, 1: on, 2: blink
//int lenMicroSecondsOfPeriod = 20 * 1000; // 20 milliseconds (ms)
//int lenMicroSecondsOfPulse; // 1.0 ms is 0 degrees

// ESP8266
SoftwareSerial work_rtx(ESP8266_RX, ESP8266_TX); // RX 10 | TX 11
ESP8266 wifi(work_rtx);
String State_Command="";
char LED_ESP8266_Status_Feedback[128];

int Now_Delay = 0;
// System
unsigned long Prev_Sys_Cnt_LED=0, Prev_Sys_Cnt_ESP8266=0, Now_Sys_Cnt=0;

void Auto_reset_ESP8266(HardwareSerial &debug_rtx, SoftwareSerial &work_rtx, uint32_t baud, String ssid, String pwd, int reset, int port);
void ESP8266_Thread();
void Servo_Position(int angle_in_deg);
void LED_State(int state);

void setup() {
  Serial.begin(9600);
  pinMode(LED_PIN, OUTPUT);
  pinMode(ESP8266_Reset_Pin, OUTPUT); digitalWrite(ESP8266_Reset_Pin, HIGH);
  delay(500);
  Auto_reset_ESP8266(Serial, work_rtx, 9600, ESP_8266_SSID, ESP_8266_PASSWORD, ESP8266_Reset_Pin, ESP_8266_HOST_PORT);
}

void loop()
{
  Now_Sys_Cnt=millis();  
  LED_State(State);
  if ((Now_Sys_Cnt-Prev_Sys_Cnt_LED)>Max_Delay_Cnt_LED){
    if(Prev_State==State){
    }
    else{
      Prev_State=State;
      if (State==0)
        Serial.println("Off!");
      else if (State==1)
        Serial.println("On!");
      else if (State==2)
        Serial.println("Blink!");
    }
    Prev_Sys_Cnt_LED=Now_Sys_Cnt;
  }
  if ((Now_Sys_Cnt-Prev_Sys_Cnt_ESP8266)>Max_Delay_Cnt_ESP8266){
    ESP8266_Thread();
    Prev_Sys_Cnt_ESP8266=Now_Sys_Cnt;
  }
  if(Serial.read()=='!'){
    Auto_reset_ESP8266(Serial, work_rtx, 9600, ESP_8266_SSID, ESP_8266_PASSWORD, ESP8266_Reset_Pin, ESP_8266_HOST_PORT);
  }
}

void Auto_reset_ESP8266(HardwareSerial &debug_rtx, SoftwareSerial &work_rtx, uint32_t baud, String ssid, String pwd, int reset_pin, int port) {

  debug_rtx.println("wifi reset...");
  delay(2000);
  pinMode(reset_pin, OUTPUT);
  digitalWrite(reset_pin, LOW);
  delay(500);
  digitalWrite(reset_pin, HIGH);
  wifi.init(baud);
  delay(500);
  if (wifi.setOprToStation()) {
    debug_rtx.print("STA_Y,");
  } else {
    debug_rtx.print("STA_N,");
  }

  if (wifi.joinAP(ssid, pwd)) {
    debug_rtx.print("AP_Y,");
    debug_rtx.print("IP: ");
    debug_rtx.println(wifi.getLocalIP().c_str());
  } else {
    debug_rtx.print("AP_N,");
  }

  if (wifi.enableMUX()) {
    debug_rtx.print("MUX_Y,");
  } else {
    debug_rtx.print("MUX_N,");
  }

  if (wifi.startTCPServer(port)) {
    debug_rtx.print("TCPSER_Y,");
  } else {
    debug_rtx.print("TCPSER_N,");
  }
  if (wifi.setTCPServerTimeout(300)) {
    debug_rtx.print("T_OUT_Y,");
  } else {
    debug_rtx.print("T_OUT_N");
  }
  debug_rtx.println("wifi ok.");
}

void ESP8266_Thread() {
  uint8_t ESP8266_buffer[14] = {0}, *data_index;
  uint8_t mux_id = 0;
  uint32_t len = 0;
  String ESP8266_Command_Info = "";
  bool Send_back_bool, releaseTCP_bool;
  String Command_Info = "";
  memset(LED_ESP8266_Status_Feedback,0,sizeof(LED_ESP8266_Status_Feedback));
  //len = wifi.recv(mux_id, ESP8266_buffer, sizeof(ESP8266_buffer), 100);
  len = wifi.recv(ESP8266_buffer, sizeof(ESP8266_buffer), 500);

  /*
  if (Now_Delay > Max_Delay) {
    if (wifi.kick() == false) {
      Auto_reset_ESP8266(Serial, work_rtx, 115200, ESP_8266_SSID, ESP_8266_PASSWORD, ESP8266_Reset_Pin, ESP_8266_HOST_PORT);
    }
    Now_Delay = 0;
  }
  else
    Now_Delay++;
  */
  if (len > 0) {
    data_index = ESP8266_buffer;
    for (uint32_t i = 0; i < len-6; i++) {
      if (*data_index == ',' && *(data_index + 6) == ',') {
        for (int j = 0; j < 7; j++) {
          Command_Info += (char)data_index[j];
        }
        break;
      }
      data_index++;
    }

    if (Command_Info == ESP8266_Command_ON) {
      State_Command = "LED Bright"; State = 1;
    }
    else if (Command_Info == ESP8266_Command_OFF){
      State_Command = "LED Off"; State = 0;
    }
    else if (Command_Info == ESP8266_Command_BLINK) {
      State_Command = "LED Blink"; State = 2;
    }
    
  }
  
    ESP8266_Command_Info = State_Command + ", Target State: " + State + "\n";
    ESP8266_Command_Info.toCharArray(LED_ESP8266_Status_Feedback, ESP8266_Command_Info.length());
    Send_back_bool = wifi.send(mux_id, (unsigned char*)LED_ESP8266_Status_Feedback, ESP8266_Command_Info.length());
  
}
void LED_State(int state)
{
    if (state == 0){
      digitalWrite(LED_PIN, LOW);
    }
    else if (state == 1){
      digitalWrite(LED_PIN, HIGH);
    }
    else if (state == 2){
      digitalWrite(LED_PIN, LOW);
      delay(1000); 
      digitalWrite(LED_PIN, HIGH);
      delay(1000);      
    }     
}

