"""Six demo patients for the doctor dashboard (shown to doctors and judges).

Pure data: no database code and no import of `backend TMS`. The seeding code
(B2) reads DEMO_CASES and stores each day through the same path /analys uses.

Each case's expect_top1 / expect_zone must match what /analys returns for every
day of the case; test_demo_seed.py checks this against the model.
"""

DEMO_CASES: list[dict] = [
    {
        "key": "demo-chest-pain",
        "name": "Ержан Сейткали",
        "city": "Астана",
        "sex": "M",
        "age": 64,
        "days": [
            {
                "minutes_ago": 14,
                "lang": "ru",
                "complaint": ("Давит за грудиной уже минут сорок, даже когда сижу и не двигаюсь. "
                              "Отдаёт в левое плечо и руку, весь в холодном поту, тошнит. "
                              "Дышать тяжело. Курю с молодости."),
                "evidences": ["E_14", "E_53", "E_57_@_V_195", "E_50", "E_66", "E_148", "E_79"],
                "absent": ["E_91"],
                "unknown": [],
                "expect_top1": "Possible NSTEMI / STEMI",
                "expect_zone": "red",
            },
        ],
    },
    {
        "key": "demo-pulmonary-embolism",
        "name": "Динара Бекова",
        "city": "Шымкент",
        "sex": "F",
        "age": 43,
        "days": [
            {
                "minutes_ago": 37,
                "lang": "ru",
                "complaint": ("Резко стала задыхаться, колет под левой лопаткой, особенно когда "
                              "глубоко вдыхаю. Левая голень опухла и болит. Сегодня кашляла с кровью. "
                              "Температуры нет."),
                "evidences": ["E_45", "E_66", "E_220", "E_53", "E_151", "E_152_@_V_120"],
                "absent": ["E_91"],
                "unknown": [],
                "expect_top1": "Pulmonary embolism",
                "expect_zone": "red",
            },
        ],
    },
    {
        "key": "demo-pneumonia",
        "name": "Нурлан Ермеков",
        "city": "Караганда",
        "sex": "M",
        "age": 57,
        "days": [
            {
                "minutes_ago": 4320,  # three days earlier
                "lang": "ru",
                "complaint": "Третий день температура, по ночам знобит, кашель. Горло не болит.",
                "evidences": ["E_91", "E_201", "E_94"],
                "absent": ["E_97"],
                "unknown": [],
                "expect_top1": "Pneumonia",
                "expect_zone": "yellow",
            },
            {
                "minutes_ago": 96,
                "lang": "ru",
                "complaint": ("Кашель стал сильнее, мокрота жёлтая и густая. Аппетита совсем нет, "
                              "всё тело ломит, температура держится. Горло не болит."),
                "evidences": ["E_91", "E_201", "E_94", "E_77", "E_161", "E_144"],
                "absent": ["E_97"],
                "unknown": [],
                "expect_top1": "Pneumonia",
                "expect_zone": "yellow",
            },
        ],
    },
    {
        "key": "demo-otitis",
        "name": "Айгерим Жумабекова",
        "city": "Павлодар",
        "sex": "F",
        "age": 23,
        "days": [
            {
                "minutes_ago": 58,
                "lang": "kk",
                "complaint": ("Сол жақ құлағым екі күннен бері қатты ауырып тұр. Қызуым 38-ге дейін "
                              "көтерілді. Мұрным бітеліп, сұйық ағады. Жөтел жоқ."),
                "evidences": ["E_91", "E_53", "E_55_@_V_130", "E_181"],
                "absent": ["E_201"],
                "unknown": [],
                "expect_top1": "Acute otitis media",
                "expect_zone": "yellow",
            },
        ],
    },
    {
        "key": "demo-allergic-rhinitis",
        "name": "Тимур Абенов",
        "city": "Алматы",
        "sex": "M",
        "age": 22,
        "days": [
            {
                "minutes_ago": 121,
                "lang": "en",
                "complaint": ("Every spring my nose gets stuffy and runny and my eyes itch like crazy. "
                              "My nose and throat are itchy too, and my mom has hay fever. "
                              "No fever, no cough, it's just the pollen."),
                "evidences": ["E_181", "E_170", "E_169", "E_86"],
                "absent": ["E_91", "E_201"],
                "unknown": [],
                "expect_top1": "Allergic sinusitis",
                "expect_zone": "green",
            },
        ],
    },
    {
        "key": "demo-common-cold",
        "name": "Жанар Оспанова",
        "city": "Петропавловск",
        "sex": "F",
        "age": 34,
        "days": [
            {
                "minutes_ago": 6,
                "lang": "ru",
                "complaint": ("Со вчерашнего дня першит в горле, нос заложен и течёт прозрачное. "
                              "Голова болит, особенно лоб. Температуры нет, кашля тоже нет."),
                "evidences": ["E_97", "E_181", "E_55_@_V_89"],
                "absent": ["E_91", "E_201"],
                "unknown": [],
                "expect_top1": "URTI",
                "expect_zone": "green",
            },
        ],
    },
]
