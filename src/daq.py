import daqhats as dq
import numpy as np
from time import sleep

# define constants
SAMPLE_RATE = 10e3 # in samples
SUBSAMPLING_TIME = 10 # in s
BUFFER_SIZE = SAMPLE_RATE*SUBSAMPLING_TIME
CHANNEL_NB = 10
CHANNELS = tuple(range(CHANNEL_NB))

def check_installed_board() -> int:
    hat_lst = dq.hat_list(filter_by_id=dq.HatIDs.MCC_128)
    # return hat_lst[0] if hat_lst else None
    try:
        return hat_lst.address
    except IndexError:
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
    except dq.HatError:
        print("No board was connected, or the connection is faulty!")
        return
    
    set_modes(board)
    return board


def calibration_routine(board:dq.mcc128) -> None: # needed?
    coeff = board.calibration_coefficient_read(a_in_range=board.a_in_range_read())
    print(coeff) # just this for now


def loop(board:dq.mcc128) -> None:
    status, buff = board.a_in_scan_read()



def check_scan_start(board:dq.mcc128) -> bool:
    scan_status = board.a_in_scan_status()
    if scan_status.hardware_overrun or scan_status.buffer_overrun:
        print("Hardware or Buffer Overrun!")
        # raise dq.HatError
    return scan_status.running


def print_scan_status(board:dq.mcc128) -> None:
    ...

def kill(board:dq.mcc128) -> None:
    board.a_in_scan_stop()
    board.a_in_scan_cleanup()
    raise Exception

#--------------------------------------#
def main(*args, **kwargs) -> None:
    board = initialize_board()
    calibration_routine(board)
    print_board_properties(board)

    board.a_in_scan_start()
    i = 0
    while True:
        sleep(5)
        loop(board)
        i += 1
        if i % SAMPLE_RATE: print_scan_status()

        if i == np.inf: break

    kill(board)






if __name__ == "__main__":
    main()