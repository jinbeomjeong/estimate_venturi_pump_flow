import time, threading, datetime, logging, struct
import onnxruntime as ort
import RPi.GPIO as GPIO
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

from pymodbus.client import ModbusSerialClient
from pymodbus.framer import FramerType

# Configure logging for the application
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Set NumPy print options for better readability
np.set_printoptions(precision=2, suppress=True)

# Global variables and their initializations
value_lock = threading.Lock()  # Lock for thread-safe access to shared variables
trigger_sig = 0  # Signal to trigger certain actions
t0_period = time.perf_counter()  # Initial time for period measurement
pressure_sensor_arr = np.zeros(shape=2, dtype=np.float32)  # Array to store pressure sensor data
rpm_indicator_arr = np.zeros(shape=1, dtype=np.float32)  # Array to store RPM data
flowrate_indicator_arr = np.zeros(shape=1, dtype=np.float32)  # Array to store flowrate data
logging_data = pd.DataFrame()  # DataFrame to accumulate data for logging
data_name_list = ['time(sec)', 'pump_inlet_pressure(bar)', 'pump_outlet_pressure(bar)',
                  'pump_speed(rpm)', 'venturi_flowrate(lpm)', 'estimation_flowrate(lpm)'] # Column names for logging data

def pressure_sensor_model(signal=0.0, mode='positive') -> float:
    """
    Calculates pressure based on an analog signal and mode.
    Args:
        signal (float): The input analog signal.
        mode (str): 'positive' or 'negative' calibration mode.
    Returns:
        float: Calculated pressure in kPa.
    """
    pressure = 0  # unit: kPa

    if mode == 'positive':
        pressure = (250 * signal) - 250
    elif mode == 'negative':
        pressure = (50.325 * signal) - 151.63

    return pressure

def bcd(d1:int, d2:int) -> int:
    """
    Combines two BCD digits into a single integer.
    Args:
        d1 (int): First BCD digit.
        d2 (int): Second BCD digit.
    Returns:
        int: Combined BCD value.
    """
    return ((d1 & 0x0F) << 8) | (d2 & 0x0F)

def value_to_reg(value:int):
    """
    Converts an integer value into a list of BCD registers.
    Args:
        value (int): The integer value to convert.
    Returns:
        list: A list of BCD register values.
    """
    v = max(0, min(99999, int(value)))
    d = [int(c) for c in f"{v:05d}"]

    return [bcd(d[0], d[1]), bcd(d[2], d[3]), bcd(d[4], 0)]

def modbus_com():
    """
    Handles Modbus communication to read sensor data (pressure, RPM, flowrate).
    Updates global arrays with the latest sensor readings.
    This function runs in a separate thread.
    """
    global pressure_sensor_arr, rpm_indicator_arr, flowrate_indicator_arr

    # Initialize Modbus serial client for sensor communication
    sensor_client = ModbusSerialClient(port="/dev/ttyAMA5", baudrate=9600, parity='N', timeout=999999)
    logger.info("modbus client initialized!")

    while True:
        # Read analog sensor data (pressure)
        analog_sensor_response = sensor_client.read_input_registers(address=0x00, count=2, device_id=1)
        analog_sensor_payload = analog_sensor_response.registers

        with value_lock:
            # Process and calibrate pressure sensor data
            pressure_sensor_arr[0] = pressure_sensor_model(analog_sensor_payload[0] / 1000, mode='negative')
            pressure_sensor_arr[0] = pressure_sensor_arr[0]+5  ## sensor calibration(unit: kpa)
            pressure_sensor_arr[0] = pressure_sensor_arr[0]/100  # unit: bar

            pressure_sensor_arr[1] = ((analog_sensor_payload[1] / 1000)*1.25)-5  # unit: bar

        # Read RPM indicator data
        rpm_indicator_1_response = sensor_client.read_input_registers(address=0x3E9, count=1, device_id=2)  # read indicate value
        rpm_indicator_1_payload = rpm_indicator_1_response.registers

        with value_lock:
            rpm_indicator_arr[0] = rpm_indicator_1_payload[0]

        # Read flowrate indicator data
        flowrate_indicator_1_response = sensor_client.read_holding_registers(address=0x500-1, count=2, device_id=4)
        flowrate_indicator_1_payload = flowrate_indicator_1_response.registers

        # Convert raw flowrate data (two 16-bit registers) into a single float
        high_byte = flowrate_indicator_1_payload[0] << 16
        int_val = high_byte | flowrate_indicator_1_payload[1]
        pack_byte = struct.pack('I', int_val)
        flowrate_indicator_1_data = struct.unpack('f', pack_byte)[0]

        with value_lock:
            flowrate_indicator_arr[0] = flowrate_indicator_1_data

def measure_period():
    """
    Measures and logs the period of a trigger signal.
    This function runs in a separate thread.
    """
    # Create a unique log file name with timestamp
    period_log_file_name = 'log_data/period_measure' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
    # Create a DataFrame for the header and write it to the log file
    measure_log_data_header = pd.DataFrame(columns=['time(sec)', 'state'])
    measure_log_data_header.to_csv(period_log_file_name, mode='a', header=True)

    while True:
        t_start = time.perf_counter()
        process_time = time.perf_counter() - t0_period

        with value_lock:
            value = trigger_sig # Read trigger signal state

        # Log the current time and trigger signal state
        log_data = pd.DataFrame(data={'time(sec)': round(process_time, 4),
                                      'state': value}, index=[0])
        log_data.to_csv(path_or_buf=period_log_file_name, mode='a', header=False)

        # Calculate sleep time to maintain a consistent logging period
        t_elapsed = time.perf_counter() - t_start
        sleep_time = 0.01 - t_elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

def write_log_data():
    """
    Writes accumulated logging data to a CSV file.
    This function runs in a separate thread.
    """
    # Create a unique log file name with timestamp
    log_file_name = 'log_data/data' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
    # Create a DataFrame for the header and write it to the log file
    log_data_header = pd.DataFrame(columns=data_name_list)
    log_data_header.to_csv(log_file_name, mode='a', header=True)

    while True:
        t_start = time.perf_counter()

        with value_lock:
            log_data = logging_data # Get the current logging data
            log_data.to_csv(path_or_buf=log_file_name, mode='a', header=False) # Append to log file

        # Calculate sleep time to maintain a consistent logging period
        t_elapsed = time.perf_counter() - t_start
        sleep_time = 0.2 - t_elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

def main_loop():
    """
    Main loop of the application.
    Initializes GPIO, MQTT, loads the ONNX model, and starts worker threads.
    Continuously reads sensor data, performs flowrate estimation, publishes data via MQTT,
    and logs data.
    """
    global trigger_sig, logging_data

    # GPIO pin assignments for LEDs
    LED0 = 20
    LED1 = 26

    # Configure RPi.GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED0, GPIO.OUT)
    GPIO.setup(LED1, GPIO.OUT)

    # Set initial LED states
    GPIO.output(LED0, GPIO.HIGH)
    GPIO.output(LED1, GPIO.HIGH)

    # MQTT broker configuration
    BROKER_ADDRESS = 'localhost'
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    # MQTT topics
    system_time_topic = "system/time"
    venturi_pump_flowrate_predict_topic = "venturi_pump/flowrate/predict"
    venturi_pump_flowrate_gt_topic = "venturi_pump/flowrate/ground_truth"

    # Load ONNX inference model
    seq_len = 20
    model = ort.InferenceSession(f'models/model_{seq_len}.onnx')
    logger.info("regression model loaded!")

    # Initialize input buffer for the ONNX model
    input_buf = np.zeros(shape=(1, seq_len, 3), dtype=np.float32)
    led_state = True
    t0 = time.perf_counter() # Start time for the main loop

    # --- Modbus client for output LED (commented out) ---
    # output_led_client = ModbusSerialClient(port='/dev/ttyS0',
    #                                        framer=FramerType.RTU,
    #                                        baudrate=19200,
    #                                        bytesize=8,
    #                                        parity='N',
    #                                        stopbits=1,
    #                                        timeout=0.5,
    #                                        retries=5)
    # output_led_client.connect()
    # output_led_client.write_register(device_id=1, address=0x0, value=1)  # zero blanking
    # logger.info("Output LED COM. initialized!")

    # Start Modbus communication thread
    modbus_com_thread = threading.Thread(target=modbus_com)
    modbus_com_thread.daemon = True
    modbus_com_thread.start()
    logger.info("modbus com thread started!")

    # Start log data writing thread
    log_data_thread = threading.Thread(target=write_log_data)
    log_data_thread.daemon = True
    log_data_thread.start()
    logger.info("log data thread started!")

    # --- Period measurement thread (commented out) ---
    #period_logger_thread = threading.Thread(target=measure_period)
    #period_logger_thread.daemon = True
    #period_logger_thread.start()
    #logger.info("period logger thread started!")

    # Connect to MQTT broker
    print("Server Connecting...")
    client.connect(host=BROKER_ADDRESS, port=1883)
    client.loop_start() # Start MQTT client loop in a background thread

    logger.info("main loop started!")

    while True:
        prv_time = time.perf_counter() # Time at the beginning of the current loop iteration

        with value_lock:
            trigger_sig = 1 # Set trigger signal
            pressure_arr = pressure_sensor_arr.copy() # Get a copy of pressure data
            rpm_arr = rpm_indicator_arr.copy() # Get a copy of RPM data
            flowrate_arr = flowrate_indicator_arr.copy() # Get a copy of flowrate data

        relative_time = prv_time - t0 # Calculate relative time from start

        # Toggle LED state
        led_state = not led_state
        if led_state:
            GPIO.output(LED0, GPIO.HIGH)
        else:
            GPIO.output(LED0, GPIO.LOW)

        # Update input buffer for the ONNX model
        input_buf = np.roll(a=input_buf, shift=-1, axis=1)
        input_buf[0, -1, :] = np.concatenate([pressure_arr, rpm_arr], axis=0)

        # Run ONNX model inference to estimate flowrate
        est_flow = np.squeeze(model.run(output_names=None, input_feed={'input': input_buf})).item()

        # If RPM is very low, set estimated flow to 0
        if rpm_arr[0] <= 10:
            est_flow = 0

        pred_output = 99999 # Placeholder for predicted output
        pred_output = np.clip(pred_output, 1, 99999) # Clip predicted output to a valid range

        # --- Write to output LED client (commented out) ---
        # output_led_client.write_registers(device_id=1, address=0x1, values=value_to_reg(int(pred_output)))

        # Publish data via MQTT
        client.publish(topic=system_time_topic, payload=struct.pack('<f', relative_time))
        client.publish(topic=venturi_pump_flowrate_predict_topic, payload=struct.pack('<f', pred_output))
        client.publish(topic=venturi_pump_flowrate_gt_topic, payload=struct.pack('<f', flowrate_arr[0].item()))

        with value_lock:
            trigger_sig = 0 # Reset trigger signal
            # Prepare data for logging
            logging_data = pd.DataFrame(data={data_name_list[0]: round(relative_time, 3),
                                              data_name_list[1]: round(pressure_arr[0].item(), 3),
                                              data_name_list[2]: round(pressure_arr[1].item(), 3),
                                              data_name_list[3]: round(rpm_arr[0].item(), 3),
                                              data_name_list[4]: round(flowrate_arr[0].item(), 3),
                                              data_name_list[5]: round(est_flow, 3)}, index=[0])

        period_time = time.perf_counter() - prv_time # Calculate time taken for current loop iteration

        # Introduce delay to maintain a consistent loop period (0.5 seconds)
        if period_time >= 0.5:
            delay_time = 0
        else:
            delay_time = 0.5 - period_time

        time.sleep(delay_time)

        # Log loop performance
        logger.info(f"main loop period: {(time.perf_counter() - prv_time) * 1000:.1f}msec")
        logger.info(f"main loop calculation period: {period_time * 1000:.1f}msec")

        # Print sensor and estimation data to console
        print(pressure_arr)
        print(rpm_arr)
        print(est_flow, flowrate_arr[0])

if __name__ == "__main__":
    main_loop()
