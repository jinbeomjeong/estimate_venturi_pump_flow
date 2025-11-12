import time, threading, datetime, logging, joblib, struct
import RPi.GPIO as GPIO
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

from pymodbus.client import ModbusSerialClient
from pymodbus.framer import FramerType


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

np.set_printoptions(precision=2, suppress=True)

value_lock = threading.Lock()
trigger_sig = 0
t0_period = time.perf_counter()
pressure_sensor_arr = np.zeros(shape=4, dtype=np.float32)
rpm_indicator_arr = np.zeros(shape=2, dtype=np.float32)
flowrate_indicator_arr = np.zeros(shape=2, dtype=np.float32)
logging_data = pd.DataFrame()
data_name_list = ['time(sec)', 'pump_1_inlet_pressure(kpa)', 'pump_1_outlet_pressure(kpa)',
                  'pump_2_inlet_pressure(kpa)', 'pump_2_outlet_pressure(kpa)',
                  'pump_1_speed(rpm)', 'pump_2_speed(rpm)',
                  'pump_1_flowrate(lpm)', 'pump_2_flowrate(lpm)']

def pressure_sensor_model(signal=0.0, mode='positive') -> float:
    pressure = 0  # unit: kPa

    if mode == 'positive':
        pressure = (250 * signal) - 250

    elif mode == 'negative':
        pressure = (50.325 * signal) - 151.63

    return pressure


def bcd(d1:int, d2:int) -> int:
    return ((d1 & 0x0F) << 8) | (d2 & 0x0F)


def value_to_reg(value:int):
    v = max(0, min(99999, int(value)))
    d = [int(c) for c in f"{v:05d}"]

    return [bcd(d[0], d[1]), bcd(d[2], d[3]), bcd(d[4], 0)]


def modbus_com():
    global pressure_sensor_arr, rpm_indicator_arr, flowrate_indicator_arr

    sensor_client = ModbusSerialClient(port="/dev/ttyAMA5", baudrate=9600, parity='N', timeout=999999)
    logger.info("modbus client initialized!")

    while True:
        analog_sensor_response = sensor_client.read_input_registers(address=0x00, count=4, device_id=1)
        analog_sensor_payload = analog_sensor_response.registers

        with value_lock:
            pressure_sensor_arr[0] = pressure_sensor_model(analog_sensor_payload[0] / 1000, mode='negative')
            pressure_sensor_arr[0] = pressure_sensor_arr[0] + 5.3  ## sensor calibration(unit: kpa)

            pressure_sensor_arr[1] = pressure_sensor_model(analog_sensor_payload[1] / 1000, mode='positive')
            pressure_sensor_arr[1] = pressure_sensor_arr[1] + 2

            pressure_sensor_arr[2] = pressure_sensor_model(analog_sensor_payload[2] / 1000, mode='negative')
            pressure_sensor_arr[2] = pressure_sensor_arr[2] + 4.4

            pressure_sensor_arr[3] = pressure_sensor_model(analog_sensor_payload[3] / 1000, mode='positive')
            pressure_sensor_arr[3] = pressure_sensor_arr[3] + 3

        # time.sleep(0.001)

        rpm_indicator_1_response = sensor_client.read_input_registers(address=0x3E9, count=1, device_id=2)  # read indicate value
        #rpm_indicator_1_response = sensor_client.read_holding_registers(address=0x09C, count=1, device_id=2)  # read baudrate

        rpm_indicator_1_payload = rpm_indicator_1_response.registers

        with value_lock:
            rpm_indicator_arr[0] = rpm_indicator_1_payload[0]
        # time.sleep(0.001)

        # rpm_indicator_2_response = sensor_client.read_input_registers(address=0x3E9, count=1, device_id=3)
        # rpm_indicator_2_payload = rpm_indicator_2_response.registers
        # rpm_indicator_arr[1] = rpm_indicator_2_payload[0]
        # time.sleep(0.001)

        # flowrate_indicator_1_response = sensor_client.read_holding_registers(address=0x500-1, count=1, device_id=4)
        # flowrate_indicator_1_data = flowrate_indicator_1_response.registers
        # flowrate_indicator_arr[0] = flowrate_indicator_1[0]
        # time.sleep(0.001)

        # flowrate_indicator_2_response = sensor_client.read_holding_registers(address=0x500-1, count=1, device_id=5)
        # flowrate_indicator_2_data = flowrate_indicator_2_response.registers
        # flowrate_indicator_arr[1] = flowrate_indicator_2[0]
        # time.sleep(0.001)


def measure_period():
    period_log_file_name = 'log_data/period_measure' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
    measure_log_data_header = pd.DataFrame(columns=['time(sec)', 'state'])
    measure_log_data_header.to_csv(period_log_file_name, mode='a', header=True)

    while True:
        t_start = time.perf_counter()
        process_time = time.perf_counter() - t0_period

        with value_lock:
            value = trigger_sig

        log_data = pd.DataFrame(data={'time(sec)': round(process_time, 4),
                                      'state': value}, index=[0])
        log_data.to_csv(path_or_buf=period_log_file_name, mode='a', header=False)

        t_elapsed = time.perf_counter() - t_start
        sleep_time = 0.01 - t_elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)


def write_log_data():
    log_file_name = 'log_data/data' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
    log_data_header = pd.DataFrame(columns=data_name_list)
    log_data_header.to_csv(log_file_name, mode='a', header=True)

    while True:
        t_start = time.perf_counter()

        with value_lock:
            log_data = logging_data
            log_data.to_csv(path_or_buf=log_file_name, mode='a', header=False)

        t_elapsed = time.perf_counter() - t_start
        sleep_time = 0.1 - t_elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)


def main_loop():
    global trigger_sig, logging_data

    LED0 = 20
    LED1 = 26

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED0, GPIO.OUT)
    GPIO.setup(LED1, GPIO.OUT)

    GPIO.output(LED0, GPIO.HIGH)
    GPIO.output(LED1, GPIO.HIGH)

    BROKER_ADDRESS = "localhost"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    system_time_topic = "system/time"
    venturi_pump_flowrate_predict_topic = "venturi_pump/flowrate/predict"
    venturi_pump_flowrate_gt_topic = "venturi_pump/flowrate/ground_truth"

    model = joblib.load('lgb_model.pkl')
    logger.info("regression model loaded!")

    input_data = np.zeros(shape=1, dtype=np.float32).reshape(1, -1)
    led_state = True
    t0 = time.perf_counter()

    output_led_client = ModbusSerialClient(port='/dev/ttyS0',
                                           framer=FramerType.RTU,
                                           baudrate=19200,
                                           bytesize=8,
                                           parity='N',
                                           stopbits=1,
                                           timeout=0.5,
                                           retries=5)
    output_led_client.connect()
    output_led_client.write_register(device_id=1, address=0x0, value=1)  # zero blanking
    logger.info("Output LED COM. initialized!")

    # start thread for modbus com
    # modbus_com_thread = threading.Thread(target=modbus_com)
    # modbus_com_thread.daemon = True
    # modbus_com_thread.start()
    # logger.info("modbus com thread started!")

    # start thread for write log data
    log_data_thread = threading.Thread(target=write_log_data)
    log_data_thread.daemon = True
    log_data_thread.start()
    logger.info("log data thread started!")

    # start thread for measure period
    #period_logger_thread = threading.Thread(target=measure_period)
    #period_logger_thread.daemon = True
    #period_logger_thread.start()
    #logger.info("period logger thread started!")

    print("Server Connecting...")
    client.connect(host=BROKER_ADDRESS, port=1883)
    client.loop_start()

    logger.info("mian loop started!")

    while True:
        prv_time = time.perf_counter()

        with value_lock:
            trigger_sig = 1
            pressure_arr = pressure_sensor_arr
            rpm_arr = rpm_indicator_arr
            flowrate_arr = flowrate_indicator_arr

        relative_time = prv_time - t0

        led_state = not led_state

        if led_state:
            GPIO.output(LED0, GPIO.HIGH)
        else:
            GPIO.output(LED0, GPIO.LOW)

        #print(pressure_sensor_arr)
        #print(rpm_indicator_arr)

        #if rpm_arr[0] > 1100:
        pred_output = model.predict(pd.DataFrame(np.array([rpm_arr[0]]), columns=['PumpSpeed(RPM)']),
                                    num_iteration=model.best_iteration_)
        pred_output = 99999
        pred_output = np.clip(pred_output, 1, 99999)
        #noise = np.random.normal(loc=0, scale=50)
        #pred_output = pred_output + noise

        output_led_client.write_registers(device_id=1, address=0x1, values=value_to_reg(int(pred_output)))

        client.publish(topic=system_time_topic, payload=struct.pack('<f', relative_time))
        client.publish(topic=venturi_pump_flowrate_predict_topic, payload=struct.pack('<f', pred_output))
        client.publish(topic=venturi_pump_flowrate_gt_topic, payload=struct.pack('<f', flowrate_arr[0].item()))

        with value_lock:
            trigger_sig = 0
            logging_data = pd.DataFrame(data={data_name_list[0]: round(relative_time, 3),
                                              data_name_list[1]: round(pressure_arr[0].item(), 3),
                                              data_name_list[2]: round(pressure_arr[1].item(), 3),
                                              data_name_list[3]: round(pressure_arr[2].item(), 3),
                                              data_name_list[4]: round(pressure_arr[3].item(), 3),
                                              data_name_list[5]: round(rpm_arr[0].item(), 3),
                                              data_name_list[6]: round(rpm_arr[1].item(), 3),
                                              data_name_list[7]: round(flowrate_arr[0].item(), 3),
                                              data_name_list[8]: round(flowrate_arr[1].item(), 3)}, index=[0])

        period_time = time.perf_counter() - prv_time

        if period_time >= 0.09:
            delay_time = 0
        else:
            delay_time = 0.09 - period_time

        time.sleep(delay_time)
        logger.info(f"main loop period: {(time.perf_counter() - prv_time) * 1000:.1f}msec")
        logger.info(f"main loop calculation period: {period_time * 1000:.1f}msec")

if __name__ == "__main__":
    main_loop()
