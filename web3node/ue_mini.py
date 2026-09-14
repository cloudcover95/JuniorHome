from off_caps import ue_boot
def mini():
    row = ue_boot("T0_home")
    row["asked"] = "UE on the mini"
    row["ran"] = False
    row["spark"] = ue_boot("T1_spark")
    return row
