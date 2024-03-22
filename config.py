front_config = {
    "name": "Front",
    "units": [
        {
            "name": "rayleighpadding",
            "mode": "padding",
            "timeline": {
                "mode": "random",
                "distribution": "rayleigh",
                "param": [1, 2, 3],
            },
            "burstsize": {
                "mode": "random",
                "distribution": "fixed",
                "param": 1,
            },
            "duration": {
                "mode": "random",
                "distribution": "infinite",
            },
            "exittrigger": "None",
        }
    ],
    "entry": "rayleighpadding",
    "budget": 3000,
}
tamaraw_config = {
    "name": "Tamaraw",
    "defense_unit": [
        {
            "name": "fixedrate",
            "mode": "tamplate",
            "param": 123,
            "exit": {"sampler": "infinite", "trigger": "None"},
        }
    ],
    "entry": "fixed",
    "budget": 3000,
}
