
import logging

def log_func(log_file):
    log = logging.getLogger()
    log.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    log.addHandler(file_handler)
    
    return log

def print_results_log(avgs, classes, log):
    sep = ""
    col1 = ":"
    lineLen = 64

    log.info("")
    log.info("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    line += "{:>15}".format("AP") + sep
    line += "{:>15}".format("AP50") + sep
    line += "{:>15}".format("AP25") + sep
    log.info(line)
    log.info("#" * lineLen)

    for (li, label_name) in enumerate(classes):
        ap_avg = avgs[label_name]["AP"]
        ap_50o = avgs[label_name]["AP50"]
        ap_25o = avgs[label_name]["AP25"]
        line = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg) + sep
        line += sep + "{:>15.3f}".format(ap_50o) + sep
        line += sep + "{:>15.3f}".format(ap_25o) + sep
        log.info(line)

    all_ap_avg = avgs["all_AP"]
    all_ap_50o = avgs["all_AP50"]
    all_ap_25o = avgs["all_AP25"]

    log.info("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg) + sep
    line += "{:>15.3f}".format(all_ap_50o) + sep
    line += "{:>15.3f}".format(all_ap_25o) + sep
    log.info(line)


def print_results(avgs, classes):
    sep = ""
    col1 = ":"
    lineLen = 64

    print("")
    print("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    line += "{:>15}".format("AP") + sep
    line += "{:>15}".format("AP50") + sep
    line += "{:>15}".format("AP25") + sep
    print(line)
    print("#" * lineLen)

    for (li, label_name) in enumerate(classes):
        ap_avg = avgs[label_name]["AP"]
        ap_50o = avgs[label_name]["AP50"]
        ap_25o = avgs[label_name]["AP25"]
        line = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg) + sep
        line += sep + "{:>15.3f}".format(ap_50o) + sep
        line += sep + "{:>15.3f}".format(ap_25o) + sep
        print(line)

    all_ap_avg = avgs["all_AP"]
    all_ap_50o = avgs["all_AP50"]
    all_ap_25o = avgs["all_AP25"]

    print("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg) + sep
    line += "{:>15.3f}".format(all_ap_50o) + sep
    line += "{:>15.3f}".format(all_ap_25o) + sep
    print(line)
    print("")