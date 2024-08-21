import daqhats as dq
import numpy as np
import pandas as pd

# define constants
CHANNEL_NB = 10
CHANNELS = tuple(range(CHANNEL_NB))

SAMPLE_RATE = 1000 # in samples
SUPERSAMPLING_TIME = 10 # in s
SAMPLES_PER_CHANNEL = SAMPLE_RATE*SUPERSAMPLING_TIME # = BUFFER_SIZE/CHANNEL_NB
BUFFER_SIZE = SAMPLES_PER_CHANNEL*CHANNEL_NB


# functions:
def get_end_date() -> pd.Timestamp:
    userin = input('When would you like to stop the recording of this data? Type -1 to never stop recording (This also means it never gets logged)')
    try:
        return pd.Timestamp(userin)
    except TypeError or ValueError:
        if userin == -1: return pd.Timestamp.now() # so that it never ends
        print('Please input a valid date!')
        get_end_date()


def check_installed_board() -> int:
    hat_lst = dq.hat_list(filter_by_id=dq.HatIDs.MCC_128)
    # return hat_lst[0] if hat_lst else None
    try:
        return hat_lst[0]
    except (IndexError, AttributeError):
        "The wrong board was connected / No board was detected"
    

def print_board_properties(board:dq.mcc128=None) -> None:
    print(f"Board Info: {board.info()}")
    print(f"Analog Input Mode: {board.a_in_mode_read()}")


def set_modes(board:dq.mcc128) -> None:
    board.trigger_mode(dq.TriggerModes.RISING_EDGE)
    board.a_in_mode_write(dq.AnalogInputMode.SE) # ?
    board.a_in_range_write(dq.AnalogInputRange.BIP_10V) #??


def initialize_board() -> dq.mcc128:
    addr = check_installed_board()
    # it is known the board connected will be the mcc128
    try:
        board = dq.mcc128(address=addr)
        # print_board_properties(board)
    except (dq.HatError, ValueError):
        print("No board was connected, or the connection is faulty!")
        raise SystemExit
    
    set_modes(board)
    return board


def calibration_routine(board:dq.mcc128) -> None: # needed?
    coeff = board.calibration_coefficient_read(a_in_range=board.a_in_range_read())
    print(coeff) # just this for now


def check_scan_start(board:dq.mcc128) -> None:
    scan_status = board.a_in_scan_status()
    if scan_status.hardware_overrun or scan_status.buffer_overrun:
        print("Hardware or Buffer Overrun!")
        # raise dq.HatError
    if not scan_status.running:
        print("Unsuccessful board start. Try again")
        kill(board)


def print_scan_status(board:dq.mcc128) -> None:
    print(f"Scan Active Channel Count: {board.a_in_scan_channel_count()}")
    scan_status = board.a_in_scan_status()
    print(f"""Scan Status: 
          \n\tRunning: {scan_status.running}
          \n\tHardware or Buffer Overrun:{scan_status.hardware_overrun or scan_status.buffer_overrun}
          \n\tAcquisition Started: {scan_status.triggered}
          \n\tSamples Available: {scan_status.samples_available}""")
    print(f"Scan Actual Rate: {board.a_in_scan_actual_rate(CHANNEL_NB, SAMPLE_RATE)}")


def kill(board:dq.mcc128) -> None:
    board.a_in_scan_stop()
    board.a_in_scan_cleanup()
    raise SystemExit("Board Killed. RIP")


def dict_to_file(d: dict) -> None:
    pd.DataFrame(d).to_csv('data/DAQ_PI_'+str(pd.Timestamp.now().date())+'.csv')


def initialize_dict() -> dict:
    d = {f'Channel {i}': [] for i in CHANNELS}
    d['Date/Time'] = []
    return d


def loop(board:dq.mcc128, d: dict) -> dict:
    buff = board.a_in_scan_read(SAMPLES_PER_CHANNEL, timeout=SUPERSAMPLING_TIME)
    d['Date/Time'].append(pd.Timestamp.now())
    for j in CHANNELS:
        # d[f'Channel {j}'].append(buff.data[j*SAMPLES_PER_CHANNEL, (j+1)*SAMPLES_PER_CHANNEL])
        d[f'Channel {j}'].append(buff.data[j::CHANNEL_NB])
        # data should be stored in an interleaved manner: 
        # ch0_sample1, ch1_sample1, ch2_sample1, ch0_sample2, ch1_sample2, ch2_sample2, ...
    return d


#--------------------------------------#
def main(*args, **kwargs) -> None:
    end_date = get_end_date()        

    board = initialize_board()
    calibration_routine(board)
    print_board_properties(board)

    data = initialize_dict()
    board.a_in_scan_start(CHANNELS, SAMPLE_RATE, BUFFER_SIZE, dq.OptionFlags.DEFAULT)

    check_scan_start(board) 
    
    while pd.Timestamp.now() != end_date:
        loop(board, data) # sleep handled here with timeout=SUPERSAMPLING_TIME
        
        print_scan_status()
    dict_to_file(data)
    kill(board)


if __name__ == "__main__":
    main()