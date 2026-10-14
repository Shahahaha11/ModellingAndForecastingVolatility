outputs_by_year_04.pkl
    contains base data + freshly estimated non-MFIV TCN columns

---------------- ABOVE RUNNING - BELOW TO RUN ---------------------------
GRU analysis:
    load _04
    estimate GRU
    append native gru_* columns
    save _05
    THEN apply GRU → TCN trader adapter only in memory
TSM analysis:
    load _05
    estimate TSM
    append native tsm_* columns
    save _06
    THEN apply TSM → TCN trader adapter only in memory

ITR analysis:
    load _06
    append native itr_* columns
    save _07

DL analysis:
    load _07
    append native dl_* columns
    save _08
---------------- ABOVE EDITING - BELOW TO EDIT ---------------------------