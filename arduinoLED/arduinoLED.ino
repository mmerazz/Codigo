/*
#define led_pin 5  // Define the LED pin

bool ledState = false;  // Track LED state

void setup() {
    pinMode(led_pin, OUTPUT);
    Serial.begin(9600);  // Match baud rate with Python
}

void loop() {
    if (Serial.available() > 0) {  // Check if data is available
        char command = Serial.read();  // Read the command from Python
        if (command == 'T') {
            ledState = !ledState;  // Toggle LED state
            digitalWrite(led_pin, ledState ? HIGH : LOW);
            Serial.println(ledState ? "LED ON" : "LED OFF");  // Print status for debugging
        }
    }
}
*/
