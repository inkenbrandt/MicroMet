"""
This module contains station-specific information, such as site folders and logger IDs.
"""

site_folders = {
    "US-UTD": "Dugout_Ranch",
    "US-UTB": "BSF",
    "US-UTJ": "Bluff",
    "US-UTW": "Wellington",
    "US-UTE": "Escalante",
    "US-UTM": "Matheson",
    "US-UTP": "Phrag",
    "US-CdM": "Cedar_mesa",
    "US-UTV": "Desert_View_Myton",
    "US-UTN": "Juab",
    "US-UTG": "Green_River",
}

loggerids = {
    "eddy": {
        "US-UTD": [21314],
        "US-UTB": [27736],
        "US-UTJ": [21020],
        "US-UTW": [21025],
        "US-UTE": [21021],
        "US-UTM": [21022, 21029],
        "US-UTP": [8442],
        "US-CdM": [21313],
        "US-UTV": [21027],
        "US-UTN": [8441],
        "US-UTG": [25415],
        "US-UTL": [21215],
    },
    "met": {
        "US-UTD": [21031],
        "US-UTB": [27736],
        "US-UTJ": [21030],
        "US-UTW": [21026],
        "US-UTE": [21032],
        "US-UTM": [21024, 21023],
        "US-UTP": [8441],
        "US-CdM": [21029],
        "US-UTV": [21311],
        "US-UTN": [],
        "US-UTG": [25414],
        "US-UTL": [21028],
    },
}
