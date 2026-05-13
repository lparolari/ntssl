rspp_ad_vs_nonad_mapping = (
    {
        "Low-grade adenoma": 1,
        "High-grade adenoma": 1,
        "Traditional serrated adenoma": 0,
        "Sessile serrated lesion": 0,
        "Hyperplastic polyp": 0,
        "Invasive cancer (T1b)": -1,
    }  # SUN
    | {
        "AD": 1,
        "SSL": 0,  # non-adenoma
        "TSA": 0,  # non-adenoma
        "HP": 0,  # non-adenoma
        "NO POLYP": -1,
        "OTHER": -1,
    }  # RC
    | {
        "Tubular Adenoma": 1,
        "Tubular Adenoma With High-Grade Dysplasia (HGD)": 1,
        "Tubulovillous Adenoma": 1,
        "Sessile Serrated Lesion (SSL)": 0,  # non-adenoma
        "Hyperplastic Polyp": 0,  # non-adenoma
        "Inflammatory Polyp": 0,  # non-adenoma
        "Pathology Not Available": -1,
    }  # POLYPSIZE
    | {
        "adenomatous": 1,
        "hyperplastic": 0,
    }  # POLYPSSET
)
