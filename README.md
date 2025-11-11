# 🧠 Manchester_Testbed_Embedded_Controller  
**An adaptive embedded control and data acquisition framework for the Manchester Ion Beam Testbed**

---

## Overview  

**Manchester_Testbed_Embedded_Controller** is a modular, intelligent control framework developed for the Manchester Ion Beam Testbed.  
It integrates an **ESP32-based embedded controller**, a **Python backend**, and an interactive **Plotly Dash dashboard** to enable real-time data acquisition, adaptive feedback, and autonomous experimental optimisation.  

The system is designed to dynamically tune control voltages in response to live measurements, leveraging machine learning (RNN-based adaptive control) to identify optimal operational parameters in real time.

---

## System Architecture  

+------------------------------------------------------+
| ESP-12F Dashboard |
| Real-time visualisation and manual control via Python package - Dash.|
+-------------------------------+----------------------+
|
▼
+------------------------------------------------------+
| Python Backend |
| Serial communication, data logging and control architectures for/of the ESP-12F Arduino Board.|
+-------------------------------+----------------------+
|
▼
+------------------------------------------------------+
| ESP32 Embedded Firmware |
| Real-time DAC/ADC control, OTA updates, diagnostics from the testbed via physical connections to the board.|
+------------------------------------------------------+

---

## Key Features  

### Embedded Firmware (C++ / Arduino Framework)
- Real-time DAC control (MCP4725) and ADC acquisition (ADS1115)  
- Serial and OTA (Wi-Fi) communication with the backend  
- Configurable logging, debugging, and LED-based status signalling  
- Modular structure for expansion and integration with new devices  

### Python Backend
- Serial communication management and data buffering  
- Automated CSV logging with timestamped autosave  
- Live plotting of voltage signals for diagnostics  
- Variance-based stability detection for adaptive control  

### Plotly Dash Dashboard
- Browser-based interface for control and monitoring.  
- Live plotting of measured voltages.
- DAC voltage adjustment via manual input boxes.  
- Real-time metrics for connections, voltages, and data feeds. 

### Adaptive Machine Learning Control (in development)
- Recurrent Neural Network (RNN) architecture  
- Incremental online learning via backpropagation  
- Variance-based detection of steady-state conditions  
- Predictive voltage adjustment for voltage optimisations  

---

## Technology Stack  

| Layer | Technologies |
|-------|---------------|
| **Firmware** | ESP32 / Arduino Framework / I²C (ADS1115, MCP4725) |
| **Backend** | Python 3.14|
| **Frontend** | Plotly Dash |
| **ML Module** | Torch / scikit-learn |

---

Contributors:
Ethan Barnes
Doctoral Researcher — University of Manchester, School of Physics & Astronomy, Nuclear Physics Department
Project: Manchester Ion Beam Testbed — Embedded Control Framework

License
Released under the MIT License.
This software is open-source and may be used, modified, and distributed with attribution and recognition.
