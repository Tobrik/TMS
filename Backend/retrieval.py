"""
Retrieval-based symptom extraction for TMS.

Architecture: Vocabulary mapping (boss) + BM25 (fallback for unknowns).

Flow: Patient text → vocabulary lookup → exact E_XX codes → RF classifier.
No LLaMA needed for extraction. LLaMA used ONLY for explanation (frontend).

The vocabulary maps Russian patient terms directly to DDXPlus evidence IDs.
BM25 is kept as a fallback for symptoms not covered by the vocabulary.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

logger = logging.getLogger("tms")

_BASE = Path(__file__).parent
_TRAINING_DIR = _BASE / "training sklearn"
_EVIDENCES_PATH = _TRAINING_DIR / "release_evidences.json"

# ─── Direct vocabulary: Russian term → DDXPlus evidence IDs ──────────────────
# Each entry: (regex_pattern, [evidence_ids]).
# Pattern matched against LOWERCASE patient text.
# No \b word boundaries — they don't work reliably with Cyrillic in Python re.
# Patterns use spacing and character anchors instead.

_VOCAB: List[Tuple[str, List[str]]] = [
    # ── Respiratory ────────────────────────────────────────────────────────────
    (r"насморк|ринит|заложен.{0,10}нос|нос.{0,10}заложен|течёт\s+нос|сопли",
     ["E_181"]),  # nasal congestion / runny nose
    (r"гнойн\w+.{0,15}выдел|жёлт\w+.{0,10}выдел|зелён\w+.{0,10}выдел|гнойн\w+\s+насморк",
     ["E_182"]),  # greenish/yellowish nasal discharge
    (r"каш\w+",
     ["E_201"]),  # cough (каш\w+ covers кашель, кашляет, кашлять)
    (r"мокрот\w+|продуктивн\w+\s+каш\w+|каш\w+\s+с\s+мокрот\w+|жёлт\w+\s+мокрот\w+|зелён\w+\s+мокрот\w+",
     ["E_77"]),   # colored/abundant sputum
    (r"боль.{0,20}при\s+вдох|больно\s+дышать|боль.{0,20}при\s+дыхани|боль.{0,20}вдох|грудь.{0,10}болит.{0,10}дыш|при\s+глубок\w+\s+вдох",
     ["E_220"]),  # pain on deep breathing (pleuritic)
    (r"одышк\w+|задыха\w+|трудно\s+дышать|нехватка\s+воздуха",
     ["E_66"]),   # shortness of breath
    (r"свист\w+.{0,10}дыхани|хрипит|хриплое\s+дыхани|свист\w+.{0,10}выдох|хрипы\s+при\s+выдох",
     ["E_214"]),  # wheezing on exhale
    (r"стридор|свист\w+.{0,10}вдох|шумн\w+\s+дыхани",
     ["E_112"]),  # stridor / wheezing on inhale
    (r"боль.{0,10}горл|горло.{0,10}бол|глотать\s+больно|больно\s+глотать|фарингит|тонзиллит|горло\s+дер\w+|болит\s+горло",
     ["E_97", "E_53", "E_55_@_V_148"]),  # sore throat → pain present + pharynx location
    (r"чихани|чихает|чихаю",
     ["E_181"]),  # sneezing

    # ── Fever / Systemic ───────────────────────────────────────────────────────
    (r"температур\w+|лихорадк\w+|горячк\w+|повышен\w+\s+температур|(?:^|\s)жар(?:\s|,|\.)",
     ["E_91"]),   # fever
    (r"озноб|дрожь|дрожит|знобит|трясёт",
     ["E_94"]),   # chills / shivers
    (r"ломот\w+|мышечн\w+\s+боль|боли\s+в\s+мышц\w+|ноет\s+всё\s+тело|боль\s+во\s+всём\s+тел",
     ["E_144"]),  # diffuse muscle pain
    (r"слабост\w+|усталост\w+|недомогани\w+|нет\s+сил|разбит\s",
     ["E_175"]),  # fatigue / malaise
    (r"потею|потливост\w+|сильно\s+потею|пот\s+льёт|испарина",
     ["E_50"]),   # increased sweating

    # ── Cardiovascular ─────────────────────────────────────────────────────────
    (r"боль.{0,10}груди|грудн\w+\s+боль|боль.{0,10}грудин|стенокарди|грудь\s+болит|сжимает\s+грудь|давит\s+в\s+груди|сжатие\s+в\s+груди",
     ["E_14", "E_53", "E_55_@_V_101"]),  # chest pain → pain present + upper chest location
    (r"сердцебиени\w+|тахикарди\w+|сердце\s+бьётся\s+быстро|учащённ\w+\s+пульс|серд\w+\s+колотится",
     ["E_155"]),  # palpitations / racing heart
    (r"головокружени\w+|кружится\s+голова|голова\s+кружится|вертиго",
     ["E_82"]),   # lightheadedness / dizziness
    (r"обморок|потерял\s+сознани|потеря\s+сознани|упал\s+в\s+обморок",
     ["E_82"]),   # near-syncope
    (r"отдаёт.{0,20}лев\w+.{0,10}(?:рук|плеч|лопатк)|боль.{0,10}лев\w+\s+рук|иррадиир\w+.{0,20}лев\w+|лев\w+.{0,10}(?:рук|плеч).{0,10}(?:боль|отдаёт)",
     ["E_14", "E_53", "E_57_@_V_195"]),  # pain radiates to left arm/shoulder
    (r"отдаёт.{0,20}(?:челюст|jaw|нижн\w+\s+челюст)|боль.{0,10}челюст|челюст\w+\s+боль",
     ["E_53", "E_57_@_V_121"]),  # pain radiates to jaw
    (r"боль\s+отдаёт|иррадиир\w+|отдаёт.{0,15}плеч",
     ["E_14", "E_53"]),  # generic radiation (unspecified direction)

    # ── Gastrointestinal ───────────────────────────────────────────────────────
    (r"тошнот\w+|тошнит|чувство\s+тошнот",
     ["E_148"]),  # nausea
    (r"рвот\w+|вырвало|стошнило|блеват|рвёт",
     ["E_148"]),  # vomiting (E_211 = REPEATED vomiting, only on explicit "несколько раз рвало")
    (r"несколько\s+раз\s+рвало|многократная\s+рвота|рвота\s+несколько\s+раз|повторная\s+рвота",
     ["E_211"]),  # repeatedly vomited
    (r"боль.{0,10}живот|живот.{0,10}бол|болит\s+живот|брюшн\w+\s+боль",
     ["E_53", "E_55_@_V_187"]),  # abdominal pain → pain present + belly location
    (r"боль.{0,15}прав\w+.{0,10}бок|боль.{0,10}справа\s+внизу|прав\w+\s+подвздошн|прав\w+.{0,10}нижн\w+.{0,10}(?:живот|бок)",
     ["E_53", "E_55_@_V_87"]),   # RLQ pain → pain present + right iliac fossa location
    (r"понос|диарея|жидкий\s+стул|частый\s+стул",
     ["E_51"]),   # diarrhea
    (r"изжога|кислый\s+привкус|кислотн\w+\s+рефлюкс",
     ["E_173", "E_53", "E_55_@_V_197", "E_54_@_V_181"]),  # GERD → pain + epigastric + burning
    (r"жжение.{0,10}груди|жжёт\s+(?:в\s+груди|за\s+грудин)|жжение\s+за\s+грудин",
     ["E_173", "E_53", "E_55_@_V_101", "E_54_@_V_181"]),  # burning in chest
    (r"давящ\w+\s+боль|тяжест\w+\s+в\s+груди|давит\s+(?:в\s+груди|на\s+грудь)|сжимающ\w+\s+боль|тяжесть.{0,10}груди",
     ["E_53", "E_54_@_V_183"]),  # pressing/heavy pain character

    # ── Head pain / headache (RU) ────────────────────────────────────────────
    (r"головн\w+\s+бол|болит\s+голов|голова\s+(?:бол|раскалыва|разламыва|гудит|трещит)|цефалги",
     ["E_53", "E_55_@_V_89"]),  # headache → pain + forehead
    (r"мигрен",
     ["E_53", "E_55_@_V_89", "E_54_@_V_184", "E_99"]),  # migraine → pain + forehead + pulsating + history
    (r"боль.{0,10}виск|висок.{0,10}бол|височн\w+\s+бол|стучит\s+в\s+виск",
     ["E_53", "E_55_@_V_166"]),  # temple pain
    (r"боль.{0,10}затылк|затылок.{0,10}бол|затылочн\w+\s+бол|ломит\s+затылок",
     ["E_53", "E_55_@_V_124"]),  # occiput pain
    (r"боль.{0,10}макушк|макушк\w+\s+бол|темечк|боль.{0,10}теменн",
     ["E_53", "E_55_@_V_62"]),  # top of head pain
    (r"(?:одн\w+\s+)?(?:сторон\w+\s+)?(?:головн\w+\s+бол|голов\w+\s+бол)|гемикрани|полголовы\s+бол|бол\w+\s+полголовы",
     ["E_53", "E_55_@_V_166"]),  # unilateral headache → temple

    # ── Pain character (RU) ──────────────────────────────────────────────────
    (r"пульсир\w+\s+бол|пульсир\w+\s+головн|бол\w+\s+пульсир|пульсаци|стучит\s+в\s+голов",
     ["E_54_@_V_184"]),  # pulsating pain
    (r"остр\w+\s+бол|резк\w+\s+бол|режущ\w+\s+бол|кинжальн",
     ["E_54_@_V_192"]),  # sharp pain
    (r"колющ\w+\s+бол|колет|прокалыва",
     ["E_54_@_V_179"]),  # stabbing / knife-like
    (r"тупая\s+бол|ноющ\w+\s+бол|ноет|нудн\w+\s+бол",
     ["E_54_@_V_154"]),  # dull / aching
    (r"схваткообразн|спазм\w+\s+бол|бол\w+\s+спазм|бол\w+\s+как\s+схватк|судорожн\w+\s+бол",
     ["E_54_@_V_182"]),  # cramping pain

    # ── Photophobia / phonophobia / aura (RU) — map to available evidences ───
    (r"светобоязн|фотофоби|свет\s+(?:раздража|бол|мешает|режет|невыносим)|(?:яркий|дневной)\s+свет.{0,10}(?:бол|невыносим|раздраж|мешает)|больно\s+смотреть\s+на\s+свет",
     ["E_127", "E_99"]),  # light sensitivity → tears(proxy) + migraine history
    (r"фонофоби|звук\w+\s+(?:раздража|мешает|невыносим|бол)|шум\s+(?:раздража|невыносим|мешает)|громк\w+\s+звук\w+\s+(?:бол|невыносим)",
     ["E_99"]),  # sound sensitivity → migraine history (best proxy)
    (r"аура|мелькани\w+\s+(?:перед|в)\s+глаз|зрительн\w+\s+нарушен|пятна\s+перед\s+глаз|вспышки\s+(?:перед|в)\s+глаз|мушки\s+перед\s+глаз|зигзаг\w+\s+(?:перед|в)\s+глаз",
     ["E_99"]),  # visual aura → migraine history

    # ── Neurological / Musculoskeletal ─────────────────────────────────────────
    (r"нет\s+запаха|потеря.{0,10}(?:запаха|обоняния)|не\s+чувств\w+\s+запах|запах\s+пропал",
     ["E_103"]),  # loss of smell/olfaction
    (r"онемени\w+|покалывани\w+|немеет|онемел",
     ["E_177"]),  # numbness / tingling

    # ── Skin ───────────────────────────────────────────────────────────────────
    (r"сыпь|высыпани\w+|крапивниц\w+|пятна\s+на\s+коже|волдыр\w+|кожн\w+\s+сыпь",
     ["E_129"]),  # skin lesion / rash
    (r"зуд|чешется|сильн\w+\s+зуд",
     ["E_129"]),  # itching

    # ── Eyes / ENT ─────────────────────────────────────────────────────────────
    (r"глаза\s+красн\w+|красн\w+\s+глаза|конъюнктивит|белки\s+красн",
     ["E_74"]),   # eye redness
    (r"лимфоузл\w+|лимфатическ\w+\s+узл|узл\w+\s+увеличен|шея.{0,15}(?:увеличен|опухл)",
     ["E_9"]),    # swollen lymph nodes

    # ── Body locations (RU) — neck, back, knee, etc. ───────────────────────
    (r"боль.{0,10}ше[еи]|шея.{0,10}бол|шейн\w+\s+бол|болит\s+шея",
     ["E_53", "E_55_@_V_26"]),  # neck pain → back of neck
    (r"боль.{0,10}(?:спин|поясниц)|спина.{0,10}бол|поясниц\w+\s+бол|болит\s+спина|прострел.{0,10}спин|радикулит|люмбаго",
     ["E_53", "E_55_@_V_40"]),  # back/lumbar pain
    (r"боль.{0,10}колен|колено.{0,10}бол|коленн\w+\s+бол|болит\s+колено|колени\s+бол",
     ["E_53", "E_55_@_V_92"]),  # knee pain
    (r"боль.{0,10}плеч|плечо.{0,10}бол|плечев\w+\s+бол|болит\s+плечо",
     ["E_53", "E_55_@_V_194"]),  # shoulder pain (R)
    (r"боль.{0,10}лопатк|лопатк\w+.{0,10}бол|между\s+лопатк",
     ["E_53", "E_55_@_V_127"]),  # scapula pain
    (r"боль.{0,10}пах|пах.{0,10}бол|паховая\s+бол",
     ["E_53", "E_55_@_V_16"]),  # groin pain

    # ── General symptoms (RU) — anxiety, sleep, irritability, etc. ────────
    (r"тревог\w+|тревожност|беспокой\w+|волнуюсь|паник\w+(?!\s+атак)",
     ["E_16"]),   # anxiety
    (r"плохой\s+сон|бессонниц|не\s+высыпаюсь|не\s+могу\s+заснуть|нарушени\w+\s+сн",
     ["E_89"]),   # non-restful sleep
    (r"раздражител|раздражён|раздражаюсь|вспыльчив|перепады\s+настроен",
     ["E_114"]),  # irritability / mood instability
    (r"набрал\s+вес|поправил\w+|прибавил\s+в\s+вес|располнел|потолстел",
     ["E_96"]),   # weight gain
    (r"(?:высок\w+\s+)?давлени\w+|гипертони|(?:^|\s)ад\s+\d|повышенн\w+\s+давлен",
     ["E_102"]),  # high blood pressure
    (r"диабет|сахарн\w+\s+диабет",
     ["E_69"]),   # diabetes
    (r"курю|курильщик|курит|курени",
     ["E_79"]),   # smoking
    (r"храп|апноэ\s+сна|остановк\w+\s+дыхани\w+\s+(?:во\s+сне|ночью)",
     ["E_23"]),   # sleep apnea
    (r"травм\w+\s+голов|удар.{0,10}голов|сотрясени",
     ["E_185"]),  # head trauma

    # ── History / Context ─────────────────────────────────────────────────────
    (r"контакт\s+с\s+больн|заразил|больн\w+\s+в\s+семье|общался\s+с\s+больн",
     ["E_41"]),   # contact with person with similar symptoms
    (r"недавно\s+(?:болел|простыл|простудил)|простыл|была\s+простуда",
     ["E_116"]),  # cold in last 2 weeks
    (r"аллерги\w+|аллергическ",
     ["E_169"]),  # allergic context
    (r"астм\w+|бронхиальн\w+\s+астм",
     ["E_123"]),  # asthma/COPD history
    (r"сердечн\w+\s+недостаточност",
     ["E_106"]),  # heart failure history

    # ══════════════════════════════════════════════════════════════════════════
    # ── KAZAKH (KK) patterns ─────────────────────────────────────────────────
    # Top-20 symptoms in Kazakh with morphological variants (\w* for suffixes).
    # ══════════════════════════════════════════════════════════════════════════

    # ── Respiratory (KK) ─────────────────────────────────────────────────────
    (r"жөтел\w*",
     ["E_201"]),  # cough (жөтел, жөтеледі, жөтелім)
    (r"қақырық\w*|ақ\s+қақырық|жасыл\s+қақырық|сары\s+қақырық",
     ["E_77"]),   # sputum (қақырық)
    (r"мұрын\w*\s+бітел\w*|мұрын\w*\s+ағ\w*|тұмау\w*",
     ["E_181"]),  # nasal congestion / runny nose
    (r"ентіг\w*|тыныс\s+ал\w*\s+қиын|дем\s+ал\w*\s+қиын|демікпе\w*",
     ["E_66"]),   # shortness of breath / asthma
    (r"тамақ\w*\s+ауыр\w*|тамағым\s+ауыр\w*|жұт\w+\s+ауыр\w*|тамақ\w*\s+ісін\w*",
     ["E_97", "E_53", "E_55_@_V_148"]),  # sore throat
    (r"түшкір\w*",
     ["E_181"]),  # sneezing

    # ── Fever / Systemic (KK) ────────────────────────────────────────────────
    (r"қызба\w*|дене\s+қызу\w*|температура\w*\s+көтеріл\w*|ыстық\w*",
     ["E_91"]),   # fever
    (r"қалтыра\w*|дірілде\w*|суық\s+тиді|суық\s+ті\w*",
     ["E_94"]),   # chills
    (r"бұлшықет\w*\s+ауыр\w*|бүкіл\s+ден\w+\s+ауыр\w*|етім\s+ауыр\w*",
     ["E_144"]),  # muscle pain
    (r"әлсіздік\w*|шаршағыш\w*|шаршадым|күшім\s+жоқ|қуатсыз\w*",
     ["E_175"]),  # fatigue / weakness
    (r"терле\w*|көп\s+терлеймін|терлеп\s+кет\w*",
     ["E_50"]),   # sweating

    # ── Cardiovascular (KK) ──────────────────────────────────────────────────
    (r"кеуде\w*\s+ауыр\w*|кеудем\s+ауыр\w*|кеуде\w*\s+қыс\w*|төс\w*\s+ауыр\w*",
     ["E_14", "E_53", "E_55_@_V_101"]),  # chest pain
    (r"жүрек\w*\s+соғ\w*|жүрегім\s+қатты\s+соғ\w*|тахикарди\w*|жүрек\w*\s+лүпілде\w*",
     ["E_155"]),  # palpitations
    (r"бас\w*\s+айнал\w*|басым\s+айнал\w*|бас\s+айналу",
     ["E_82"]),   # dizziness
    (r"есінен\s+тан\w*|естен\s+тан\w*|талып\s+қал\w*",
     ["E_82"]),   # syncope

    # ── Head / Neuro (KK) ────────────────────────────────────────────────────
    (r"бас\w*\s+ауыр\w*|басым\s+ауыр\w*|бас\s+ауру\w*",
     ["E_53", "E_55_@_V_89"]),  # headache
    (r"жүйке\w*\s+ауыр\w*|ұйқысыздық\w*|ұйықтай\s+алма\w*",
     ["E_89"]),   # insomnia

    # ── Gastrointestinal (KK) ────────────────────────────────────────────────
    (r"жүрек\w*\s+айн[уы]\w*|жүрегім\s+айн\w*|қалжыра\w*|лоқсы\w*",
     ["E_148"]),  # nausea
    (r"құс\w*|құстым|қайтар\w*",
     ["E_148"]),  # vomiting
    (r"іш\w*\s+ауыр\w*|ішім\s+ауыр\w*|құрсақ\w*\s+ауыр\w*",
     ["E_53", "E_55_@_V_187"]),  # abdominal pain
    (r"іш\w*\s+өт\w+|диарея\w*|сұйық\s+нәжіс",
     ["E_51"]),   # diarrhea

    # ── Skin (KK) ────────────────────────────────────────────────────────────
    (r"бөртпе\w*|терід\w+\s+бөрт\w*|бөрту\w*",
     ["E_129"]),  # rash
    (r"қышы\w+|қышиды|қышима\w*",
     ["E_129"]),  # itching

    # ══════════════════════════════════════════════════════════════════════════
    # ── ENGLISH patterns ─────────────────────────────────────────────────────
    # Mirrors the Russian vocab above for English-language input.
    # ══════════════════════════════════════════════════════════════════════════

    # ── Head / Neuro (EN) ─────────────────────────────────────────────────────
    (r"headache|head\s*ache|head.{0,5}(?:hurts|pain|pounding|throbbing)",
     ["E_53", "E_55_@_V_89"]),  # headache → pain + forehead/head location
    (r"migraine",
     ["E_53", "E_55_@_V_89", "E_54_@_V_184", "E_99"]),  # migraine → pain + forehead + pulsating + history
    (r"visual\s+disturbance|blurr\w+\s+vision|see\w*\s+spots",
     ["E_53"]),   # visual symptoms with pain context
    (r"dizz\w+|lightheaded|vertigo|feel\w*\s+faint",
     ["E_82"]),   # dizziness
    (r"faint\w*|lost\s+consciousness|pass\w*\s+out|black\w*\s+out",
     ["E_82"]),   # syncope
    (r"numb\w+|tingl\w+|pins\s+and\s+needles",
     ["E_177"]),  # numbness / tingling

    # ── Respiratory (EN) ──────────────────────────────────────────────────────
    (r"cough\w*",
     ["E_201"]),  # cough
    (r"sputum|phlegm|cough\w*\s+(?:up|with)\s+(?:green|yellow|colored)",
     ["E_77"]),   # colored sputum
    (r"runny\s+nose|stuffy\s+nose|nasal\s+congestion|blocked\s+nose|sneezing",
     ["E_181"]),  # nasal congestion
    (r"green\w*\s+(?:nasal|discharge)|yellow\w*\s+(?:nasal|discharge)",
     ["E_182"]),  # purulent nasal discharge
    (r"shortness\s+of\s+breath|short\s+of\s+breath|can'?t\s+breathe|difficult\w*\s+breath\w*|breathless|dyspnea|hard\s+to\s+breathe",
     ["E_66"]),   # shortness of breath
    (r"wheez\w+|wheezing\s+(?:on|when)\s+exhal",
     ["E_214"]),  # wheezing on exhale
    (r"stridor|noisy\s+breath\w*\s+(?:in|when\s+inhal)",
     ["E_112"]),  # stridor
    (r"sore\s+throat|throat\s+(?:hurts|pain|ache)|painful\s+(?:to\s+)?swallow|difficulty\s+swallow",
     ["E_97", "E_53", "E_55_@_V_148"]),  # sore throat → pain + pharynx
    (r"pain.{0,15}(?:deep\s+breath|breath\w+\s+in|inhal)|hurts\s+(?:to|when\s+(?:i\s+)?)\s*breath|pleuritic|(?:hurts|pain).{0,10}breath\w*\s+(?:in\s+)?deeply",
     ["E_220"]),  # pleuritic pain

    # ── Fever / Systemic (EN) ─────────────────────────────────────────────────
    (r"fever|temperature\s+\d|feeling\s+hot|febrile",
     ["E_91"]),   # fever
    (r"chills?|shiver\w*|rigor",
     ["E_94"]),   # chills
    (r"muscle\s+(?:pain|ache)|body\s+ache|aching\s+all\s+over",
     ["E_144"]),  # muscle pain
    (r"fatigue\w*|tired|exhausted|weak\b|weakness|malaise|no\s+energy",
     ["E_175"]),  # fatigue
    (r"sweat\w+|perspir\w+|drenched\s+in\s+sweat",
     ["E_50"]),   # sweating

    # ── Cardiovascular (EN) ───────────────────────────────────────────────────
    (r"chest\s+pain|pain\s+in\s+(?:my\s+)?chest|chest\s+(?:hurts|pressure|tightness|squeezing)",
     ["E_14", "E_53", "E_55_@_V_101"]),  # chest pain → pain + upper chest
    (r"palpitat\w+|heart\s+(?:racing|pounding|beating\s+fast|flutter)|irregular\w*\s+(?:heart|pulse|beat)|(?:heart|pulse)\s+(?:is\s+)?irregular|beating\s+(?:fast\s+)?(?:and\s+)?irregular",
     ["E_155"]),  # palpitations
    (r"radiat\w*.{0,15}(?:left\s+arm|left\s+shoulder)|pain.{0,10}left\s+arm|left\s+arm.{0,10}(?:pain|numb|tingl)",
     ["E_14", "E_53", "E_57_@_V_195"]),  # radiates to left arm
    (r"radiat\w*.{0,15}jaw|jaw\s+pain|(?:and|to)\s+(?:my\s+)?jaw|pain.{0,15}jaw",
     ["E_53", "E_57_@_V_121"]),  # radiates to jaw
    (r"pressing\s+(?:pain|chest)|heavy.{0,10}chest|crushing\s+(?:pain|chest)|squeezing\s+(?:pain|chest)",
     ["E_53", "E_54_@_V_183"]),  # pressing/heavy character

    # ── Gastrointestinal (EN) ─────────────────────────────────────────────────
    (r"nausea\w*|nauseous|feel\w*\s+sick|queasy",
     ["E_148"]),  # nausea
    (r"vomit\w*|threw\s+up|throwing\s+up|thrown\s+up|puk\w+",
     ["E_148"]),  # vomiting
    (r"(?:threw|throwing|thrown)\s+up\s+(?:twice|several|multiple|many|[2-9])|vomit\w*\s+(?:twice|several|multiple|repeatedly|[2-9])",
     ["E_211"]),  # repeated vomiting
    (r"(?:stomach|belly|abdomen|abdominal)\s+(?:pain|ache|hurts|cramp)|pain\s+in\s+(?:my\s+)?(?:stomach|belly|abdomen)|(?:my\s+)?(?:stomach|belly|tummy)\s+hurts|belly\s+button|navel|umbilical|peri.?umbilical",
     ["E_53", "E_55_@_V_187"]),  # abdominal pain → pain + belly
    (r"(?:lower\s+)?right\s+(?:side|quadrant|abdomen)|right\s+lower\s+(?:side|quadrant|abdomen)|right\s+(?:side|flank).{0,10}(?:pain|hurts)|(?:pain|hurts).{0,10}(?:lower\s+)?right\s+(?:side|quadrant|abdomen)|(?:moved|migrated).{0,15}(?:lower\s+)?right\s+side",
     ["E_53", "E_55_@_V_87"]),  # RLQ pain → right iliac fossa (appendicitis)
    (r"diarr\w+|loose\s+stool|watery\s+stool|frequent\s+(?:stool|bowel)",
     ["E_51"]),   # diarrhea
    (r"heartburn|acid\s+reflux|burning.{0,10}(?:stomach|chest|throat)",
     ["E_173", "E_53", "E_55_@_V_197", "E_54_@_V_181"]),  # GERD → burning + epigastric

    # ── Skin (EN) ─────────────────────────────────────────────────────────────
    (r"rash|skin\s+(?:rash|lesion|spots)|hives|urticaria",
     ["E_129"]),  # rash
    (r"itch\w+|itchy|scratching",
     ["E_129"]),  # itching

    # ── Eyes / ENT (EN) ───────────────────────────────────────────────────────
    (r"red\s+eye|eye\w*\s+(?:red|bloodshot)|conjunctivit",
     ["E_74"]),   # eye redness
    (r"swollen\s+(?:lymph|gland)|lymph\s+node|enlarged\s+(?:lymph|gland|node)",
     ["E_9"]),    # swollen lymph nodes
    (r"(?:lost|loss).{0,10}(?:smell|taste)|can'?t\s+smell",
     ["E_103"]),  # loss of smell

    # ── Head locations (EN) — temple, occiput, top of head ──────────────────
    (r"temple\s+(?:pain|ache|headache)|pain\s+(?:in|at)\s+(?:my\s+)?temple|temporal\s+(?:pain|headache)",
     ["E_53", "E_55_@_V_166"]),  # temple pain
    (r"(?:back|base)\s+of\s+(?:my\s+)?head|occiput\w*\s+(?:pain|headache)|occipital",
     ["E_53", "E_55_@_V_124"]),  # occiput pain
    (r"top\s+of\s+(?:my\s+)?head|(?:pain|ache)\s+(?:on\s+)?top\s+of\s+head|vertex\s+(?:pain|headache)",
     ["E_53", "E_55_@_V_62"]),  # top of head pain
    (r"one\s+side.{0,10}head|half\s+(?:of\s+)?(?:my\s+)?head|unilateral\s+headache|hemicrani",
     ["E_53", "E_55_@_V_166"]),  # unilateral headache → temple

    # ── Migraine-specific (EN) ───────────────────────────────────────────────
    (r"(?:history\s+of|had|get|suffer\w*\s+from)\s+migraine|migraine\s+(?:runs|in\s+(?:my\s+)?family)|diagnosed\s+(?:with\s+)?migraine|chronic\s+migraine",
     ["E_99"]),   # migraine history
    (r"light\s+sensitiv|photophob|(?:bright\s+)?light\s+(?:hurts|bother|makes?\s+(?:it\s+)?worse)|eyes?\s+(?:hurt|sensitive).{0,10}light",
     ["E_127", "E_99"]),  # light sensitivity → tears(proxy) + migraine
    (r"sound\s+sensitiv|phonophob|noise\s+(?:hurts|bother|makes?\s+(?:it\s+)?worse)|sensitiv\w+\s+to\s+(?:noise|sound)",
     ["E_99"]),   # sound sensitivity → migraine
    (r"visual\s+aura|seeing\s+(?:zigzag|flash|spark|spot)|flash\w*\s+(?:of\s+)?light.{0,10}(?:before|eye)|scotoma",
     ["E_99"]),   # visual aura → migraine

    # ── Pain character (EN) ──────────────────────────────────────────────────
    (r"throb\w+\s+(?:pain|head|ache)|puls\w+\s+(?:pain|head|ache)|pounding\s+(?:pain|head|ache)|(?:pain|head|ache)\s+(?:is\s+)?(?:throbbing|pulsating|pounding)",
     ["E_54_@_V_184"]),  # pulsating
    (r"sharp\s+(?:pain|stab)|stabbing\s+(?:pain|sensation)|knife.?like\s+pain|lancinating",
     ["E_54_@_V_192"]),  # sharp
    (r"dull\s+(?:pain|ache)|aching\s+(?:pain|sensation)|(?:pain|ache)\s+(?:is\s+)?dull",
     ["E_54_@_V_154"]),  # dull / tedious
    (r"cramp\w+\s+(?:pain|sensation)|cramping|spasm\w*\s+(?:pain|of\s+pain)",
     ["E_54_@_V_182"]),  # cramping

    # ── Body locations (EN) ──────────────────────────────────────────────────
    (r"neck\s+(?:pain|ache|hurts|stiff)|stiff\s+neck|pain\s+in\s+(?:my\s+)?neck",
     ["E_53", "E_55_@_V_26"]),  # neck pain
    (r"(?:back|lower\s+back|lumbar)\s+(?:pain|ache|hurts)|pain\s+in\s+(?:my\s+)?(?:back|lower\s+back)|lumbago|sciatica",
     ["E_53", "E_55_@_V_40"]),  # back/lumbar pain
    (r"knee\s+(?:pain|ache|hurts)|pain\s+in\s+(?:my\s+)?knee",
     ["E_53", "E_55_@_V_92"]),  # knee pain
    (r"shoulder\s+(?:pain|ache|hurts)|pain\s+in\s+(?:my\s+)?shoulder",
     ["E_53", "E_55_@_V_194"]),  # shoulder pain
    (r"groin\s+(?:pain|ache|hurts)|pain\s+in\s+(?:my\s+)?groin",
     ["E_53", "E_55_@_V_16"]),  # groin pain

    # ── General symptoms (EN) ────────────────────────────────────────────────
    (r"anxi\w+|(?:feel\w*\s+)?nervous|worried|restless|(?:feel\w*\s+)?uneasy",
     ["E_16"]),   # anxiety
    (r"(?:can'?t|trouble|difficulty|hard\s+time)\s+sleep\w*|insomnia|sleep\s+(?:poorly|badly|terrible)|(?:non|un).?restful\s+sleep",
     ["E_89"]),   # non-restful sleep
    (r"irritabl\w+|mood\s+swing|(?:very\s+)?(?:moody|cranky|snappy|short.?tempered)",
     ["E_114"]),  # irritability
    (r"gain\w*\s+weight|(?:put|putting)\s+on\s+weight|weight\s+gain",
     ["E_96"]),   # weight gain
    (r"high\s+blood\s+pressure|hypertension|(?:bp|blood\s+pressure)\s+(?:is\s+)?(?:high|elevated)",
     ["E_102"]),  # high blood pressure
    (r"diabet\w+|(?:type\s+)?[12]\s+diabet\w*|(?:high|elevated)\s+(?:blood\s+)?sugar",
     ["E_69"]),   # diabetes
    (r"(?:i\s+)?smok\w+|(?:am\s+a\s+)?smoker|(?:pack|cigarette)\w*\s+(?:a|per)\s+day",
     ["E_79"]),   # smoking
    (r"sleep\s+apnea|snor\w+\s+(?:loudly|heavily)|stop\w*\s+breath\w+\s+(?:while|during|in)\s+sleep",
     ["E_23"]),   # sleep apnea
    (r"head\s+(?:trauma|injury)|concussion|hit\s+(?:my\s+)?head",
     ["E_185"]),  # head trauma

    # ── History / Context (EN) ────────────────────────────────────────────────
    (r"contact\w*\s+(?:with\s+)?(?:sick|ill|infected)|(?:someone|person).{0,10}(?:sick|ill)\s+(?:around|near|at)",
     ["E_41"]),   # contact with sick person
    (r"allerg\w+",
     ["E_169"]),  # allergy
    (r"asthma|(?:history\s+of\s+)?asthma",
     ["E_123"]),  # asthma history
    (r"heart\s+failure",
     ["E_106"]),  # heart failure history

    # ══════════════════════════════════════════════════════════════════════════
    # ── DISEASE-SPECIFIC patterns (EN+RU) — full 49-disease coverage ─────────
    # Each block targets symptoms pathognomonic for specific DDXPlus diseases.
    # ══════════════════════════════════════════════════════════════════════════

    # ── Ear pain → Acute otitis media ─────────────────────────────────────────
    (r"ear\s*(?:ache|pain|hurts|infection)|pain\s+in\s+(?:my\s+)?ear|otitis",
     ["E_53", "E_55_@_V_129"]),  # ear pain → pain + ear(R)
    (r"боль.{0,10}ух[еоа]|ух[оа].{0,10}бол|отит|стреляет\s+в\s+ухе|болит\s+ухо",
     ["E_53", "E_55_@_V_129"]),  # боль в ухе

    # ── Hoarse voice → Acute laryngitis ───────────────────────────────────────
    (r"hoarse|lost\s+(?:my\s+)?voice|voice\s+(?:gone|hoarse|raspy|deeper)|laryngitis|can'?t\s+speak",
     ["E_212"]),
    (r"охрип|осип|потерял\w*\s+голос|голос\s+(?:сел|пропал|хриплый)|ларингит|не\s+могу\s+говорить",
     ["E_212"]),

    # ── Sinus pain → Rhinosinusitis (acute/chronic/allergic) ──────────────────
    (r"sinus\w*|facial\s+(?:pain|pressure)|pain.{0,10}(?:forehead|cheek|around\s+(?:nose|eyes))|stuffed\s+up",
     ["E_53", "E_55_@_V_89", "E_181"]),  # sinus pain → forehead + congestion
    (r"гайморит|синусит|боль.{0,10}пазух|давит.{0,10}лоб|давит.{0,10}переносиц|тяжесть.{0,10}лиц",
     ["E_53", "E_55_@_V_89", "E_181"]),

    # ── Cluster headache — severe unilateral + eye ────────────────────────────
    (r"cluster\s+headache|excruciating.{0,15}headache|headache.{0,15}(?:eye|behind.{0,5}eye)|pain.{0,10}(?:behind|around|near)\s+(?:my\s+)?eye|(?:eye\s+water|tearing|red\s+eye).{0,15}headache",
     ["E_53", "E_55_@_V_125", "E_127"]),  # headache + eye + tearing
    (r"кластерн\w+\s+(?:головн|боль)|невыносим\w+\s+боль.{0,10}глаз|боль.{0,10}глаз.{0,10}слёз",
     ["E_53", "E_55_@_V_125", "E_127"]),

    # ── Panic attack — fear of dying + palpitations + tingling ────────────────
    (r"panic\s+attack|fear\s+(?:of\s+)?dying|afraid.{0,10}(?:die|dying|death)|feel\w*\s+like\s+(?:i'?m\s+)?dying",
     ["E_111", "E_155"]),
    (r"(?:паническ|панич)\w+\s+атак|страх\s+смерти|боюсь\s+умереть|кажется.{0,10}умира",
     ["E_111", "E_155"]),
    (r"depersonaliz|dereali|detach\w*\s+from\s+(?:body|reality|surroundings)",
     ["E_171"]),  # depersonalization
    (r"деперсонализ|дереализ|отстранён.{0,10}тел",
     ["E_171"]),

    # ── Whooping cough — paroxysmal cough + vomiting after cough ──────────────
    (r"whooping\s+cough|pertussis|paroxysm\w*\s+cough|cough\w*\s+(?:fits?|spells?|attacks?)",
     ["E_203", "E_201"]),  # intense coughing fits
    (r"коклюш|приступ\w+\s+каш\w+|каш\w+\s+(?:приступ|до\s+рвот)",
     ["E_203", "E_201"]),
    (r"vomit\w*\s+after\s+cough|cough\w*\s+(?:until|till|then)\s+(?:i\s+)?(?:vomit|threw|throw)",
     ["E_166"]),  # vomiting after cough
    (r"рвот\w+\s+после\s+каш\w+|каш\w+.{0,10}до\s+рвот",
     ["E_166"]),

    # ── Hemoptysis → TB / Pulmonary neoplasm / PE ─────────────────────────────
    (r"cough\w*\s+(?:up\s+)?blood|blood\w*\s+(?:in\s+)?(?:sputum|phlegm)|hemoptysis",
     ["E_45"]),
    (r"каш\w+\s+(?:с\s+)?кров|кров\w+\s+(?:мокрот|в\s+мокрот)|кровохаркан",
     ["E_45"]),

    # ── Weight loss → TB / Pulmonary neoplasm / Pancreatic neoplasm / HIV ─────
    (r"(?:lost|losing)\s+weight|weight\s+loss|unintentional\w*\s+(?:weight|lost)|getting\s+thin",
     ["E_162"]),
    (r"похудел|потеря\s+веса|теряю\s+вес|сильно\s+похудел|худею",
     ["E_162"]),

    # ── Night sweats → TB / HIV ───────────────────────────────────────────────
    (r"night\s+sweat|sweat\w*\s+(?:at\s+)?night|wake\s+up\s+(?:drenched|soaked|sweating)",
     ["E_50", "E_219"]),  # sweating + worse at night
    (r"ночн\w+\s+пот|потею\s+ночью|просыпаюсь\s+(?:в\s+поту|мокрый)",
     ["E_50", "E_219"]),

    # ── Pale skin → Anemia ────────────────────────────────────────────────────
    (r"(?:look|looking|appear)\s+pale|pale\s+skin|skin\s+(?:is\s+)?pale|pallor|very\s+pale",
     ["E_154"]),
    (r"бледн\w+\s+кож|кожа\s+бледн|бледност|побледнел",
     ["E_154"]),

    # ── Black stool → GI bleeding / Anemia ────────────────────────────────────
    (r"black\s+stool|tarry\s+stool|melena|stool\w*\s+(?:like\s+)?(?:coal|tar)",
     ["E_140"]),
    (r"чёрн\w+\s+стул|стул\s+(?:как\s+)?(?:уголь|дёготь)|мелена",
     ["E_140"]),

    # ── Blood in stool → GI bleeding ─────────────────────────────────────────
    (r"blood\s+in\s+(?:my\s+)?stool|bloody\s+stool|rectal\s+bleed|blood\s+(?:when|after)\s+(?:i\s+)?(?:wipe|poo|poop)",
     ["E_179"]),
    (r"кров\w+\s+(?:в\s+)?стул|кров\w+\s+из\s+(?:прямой|задн)|ректальн\w+\s+кровотечен",
     ["E_179"]),

    # ── Vomiting blood → Boerhaave / upper GI bleed ──────────────────────────
    (r"vomit\w*\s+blood|blood\w*\s+(?:in\s+)?vomit|coffee\s+ground|hematemesis|threw\s+up\s+blood",
     ["E_210"]),
    (r"рвот\w+\s+(?:с\s+)?кров|кров\w+\s+(?:в\s+)?рвот|кофейн\w+\s+гущ",
     ["E_210"]),

    # ── Difficulty swallowing → Epiglottitis / Pharyngitis ────────────────────
    (r"difficulty\s+swallow|can'?t\s+swallow|painful\s+(?:to\s+)?swallow|dysphagia|trouble\s+swallow",
     ["E_65"]),
    (r"трудно\s+глотать|не\s+могу\s+глотать|дисфаги|нарушен\w+\s+глотан",
     ["E_65"]),

    # ── Drooling / excess saliva → Epiglottitis / Dystonic reactions ──────────
    (r"drool\w*|excess\w*\s+saliva|can'?t\s+swallow\s+saliva|saliva\s+dripping",
     ["E_190"]),
    (r"слюнотечени|слюна\s+течёт|не\s+могу\s+глотать\s+слюн|обильн\w+\s+слюн",
     ["E_190"]),

    # ── Muscle weakness / paralysis → Guillain-Barré / Myasthenia / SLE ──────
    (r"(?:weak|weakness)\s+(?:in\s+)?(?:both\s+)?(?:legs?|arms?|limbs?)|ascending\s+weakness|can'?t\s+(?:move|lift|stand|walk)",
     ["E_84"]),
    (r"слабость\s+в\s+(?:обеих\s+)?(?:ногах|руках|конечност)|не\s+могу\s+(?:встать|ходить|поднять)|восходящ\w+\s+слабост",
     ["E_84"]),

    # ── Foot numbness → Guillain-Barré ────────────────────────────────────────
    (r"numb\w*\s+(?:in\s+)?(?:my\s+)?(?:feet|foot|toes)|(?:feet|foot|toes)\s+(?:are\s+)?numb|tingl\w+\s+(?:in\s+)?(?:feet|foot|toes)",
     ["E_93"]),
    (r"онемени\w+\s+(?:в\s+)?(?:стоп|ног|пальц\w+\s+ног)|ступни\s+немеют|ноги\s+онемели",
     ["E_93"]),

    # ── Facial weakness → Guillain-Barré / Myasthenia ─────────────────────────
    (r"facial\s+(?:weakness|droop|paralysis)|(?:face|mouth)\s+(?:droop|asymmetr)|one\s+side\s+of\s+(?:my\s+)?face",
     ["E_156"]),
    (r"слабость.{0,10}лиц|лицо\s+(?:перекосил|ослабл)|паралич\s+лиц|одна\s+сторона\s+лица",
     ["E_156"]),

    # ── Drooping eyelid → Myasthenia gravis ───────────────────────────────────
    (r"droop\w*\s+eyelid|eyelid\s+droop|ptosis|can'?t\s+(?:open|raise|lift)\s+(?:my\s+)?(?:eye|eyelid)|heavy\s+eyelid",
     ["E_172"]),
    (r"опущен\w+\s+век|веко\s+(?:опустил|не\s+поднимается)|птоз|тяжёл\w+\s+век",
     ["E_172"]),

    # ── Muscle spasms / neck stiffness → Dystonic reactions ───────────────────
    (r"muscle\s+spasm|neck\s+(?:stiff|spasm|rigid|lock)|can'?t\s+(?:turn|move)\s+(?:my\s+)?(?:head|neck)|torticollis|jaw\s+(?:lock|stuck|clench)",
     ["E_192", "E_193"]),
    (r"спазм\w+\s+мышц|шея\s+(?:свело|не\s+поворачивается|заклинил)|кривошея|судорог\w+\s+(?:лиц|шеи|мышц)|челюсть\s+(?:свело|заклинил)",
     ["E_192", "E_193"]),

    # ── Tongue protrusion → Dystonic reactions ────────────────────────────────
    (r"tongue\s+(?:sticking\s+out|protrud|involuntar)|can'?t\s+(?:keep|control)\s+(?:my\s+)?tongue",
     ["E_168"]),
    (r"язык\s+(?:вывалива|высовыва|не\s+могу\s+убрать)|непроизвольн\w+\s+(?:движени|высовыван)\w+\s+язык",
     ["E_168"]),

    # ── Jaw difficulty → Dystonic reactions / Epiglottitis ────────────────────
    (r"can'?t\s+open\s+(?:my\s+)?mouth|jaw\s+(?:won'?t|can'?t|stuck|locked|clench)|mouth\s+(?:stuck|locked|won'?t\s+open)|trismus",
     ["E_205"]),
    (r"не\s+могу\s+открыть\s+рот|челюсть\s+(?:заклинил|не\s+открывается)|тризм|рот\s+не\s+открывается",
     ["E_205"]),

    # ── Bloating → GI / GERD ─────────────────────────────────────────────────
    (r"bloat\w*|(?:belly|stomach|abdomen)\s+(?:distend|swell|swollen|puffed)|feel\w*\s+(?:full|stuffed)",
     ["E_30"]),
    (r"вздути\w+|живот\s+(?:раздул|надул|распир)|пучит|метеоризм",
     ["E_30"]),

    # ── Symptoms worse after eating → GERD ────────────────────────────────────
    (r"worse\s+after\s+eat|(?:symptoms?|pain)\s+(?:after|following)\s+(?:eat|meal|food)|eating\s+makes\s+it\s+worse",
     ["E_215"]),
    (r"хуже\s+после\s+еды|усиливается\s+после\s+(?:еды|приёма\s+пищи)|после\s+еды\s+(?:хуже|болит|тошнит)",
     ["E_215"]),

    # ── Worse lying down, better sitting → GERD / Pericarditis ────────────────
    (r"worse\s+(?:lying|laying)\s+down|better\s+(?:when\s+)?sitting|(?:symptoms?|pain)\s+(?:lying|lay|recumb)",
     ["E_217"]),
    (r"хуже\s+лёжа|лучше\s+(?:когда\s+)?сижу|усиливается\s+в\s+положении\s+лёж",
     ["E_217"]),

    # ── Pain improves leaning forward → Pericarditis ──────────────────────────
    (r"better\s+(?:when\s+)?(?:i\s+)?lean\w*\s+forward|(?:pain|chest)\s+(?:ease|improv|better).{0,10}lean|lean\w*\s+forward.{0,10}(?:help|ease|better|relief)|lean\s+forward",
     ["E_33"]),
    (r"легче\s+(?:когда\s+)?наклон\w+\s+вперёд|боль\s+уменьша\w+\s+(?:при\s+)?наклон|наклон\w+\s+вперёд.{0,10}(?:легче|лучше|помогает)",
     ["E_33"]),

    # ── Worse with exertion → Stable angina / COPD ───────────────────────────
    (r"worse\s+(?:with\s+)?(?:exert|exercis|effort|walking|stairs)|(?:pain|symptoms?)\s+(?:on|with|during)\s+(?:exert|effort|physical)",
     ["E_218"]),
    (r"хуже\s+при\s+(?:нагрузк|ходьб|подъём)|усиливается\s+(?:при\s+)?(?:физическ|нагрузк)|при\s+нагрузке\s+(?:хуже|больше|сильн)",
     ["E_218"]),

    # ── Pain worse with movement → Musculoskeletal / Rib fracture ─────────────
    (r"(?:pain|hurt)\s+(?:worse\s+)?(?:when\s+)?(?:i\s+)?(?:move|moving|turn|twist)|movement\s+makes\s+(?:it\s+)?worse",
     ["E_216"]),
    (r"боль\s+(?:усиливается\s+)?при\s+движении|больно\s+(?:поворачиват|двигат)|при\s+движении\s+(?:хуже|больн)",
     ["E_216"]),

    # ── Worse with coughing/straining → Pneumothorax / Hernia / Rib fracture ─
    (r"worse\s+(?:when\s+)?(?:i\s+)?(?:cough|strain|lift|sneez|bear\w*\s+down)|(?:pain|hurt)\w*\s+(?:when\s+)?(?:i\s+)?(?:cough|sneez|strain|lift)|(?:cough|sneez|strain).{0,10}(?:makes?\s+(?:it\s+)?worse|hurts)",
     ["E_221"]),
    (r"хуже\s+(?:при\s+)?(?:кашл|чихани|натуживани|поднятии)|больно\s+(?:кашлять|чихать|тужиться)|при\s+каш\w+\s+(?:хуже|больн)",
     ["E_221"]),

    # ── Worsening over 2 weeks → Stable angina / Progressive conditions ──────
    (r"(?:getting|been)\s+worse.{0,15}(?:week|day|past\s+few)|(?:symptoms?|condition)\s+(?:progress|worsen)\w*",
     ["E_13"]),
    (r"ухудшается.{0,10}(?:недел|дн)|становится\s+хуже|симптомы\s+прогрессир",
     ["E_13"]),

    # ── Choking / suffocation episodes → Laryngospasm / Panic ─────────────────
    (r"chok\w+\s+(?:episode|spell|attack|sensation)|felt\s+like\s+(?:i\s+was\s+)?choking|sudden\w*\s+(?:choking|suffocating|couldn'?t\s+breathe)",
     ["E_75"]),
    (r"приступ\s+удушь|чувство\s+(?:удушь|нехватки\s+воздуха)|задыхаюсь\s+приступ|внезапно\s+(?:перестал|не\s+мог)\s+дышать",
     ["E_75"]),

    # ── Brief suffocation resolved → Laryngospasm ────────────────────────────
    (r"couldn'?t\s+breathe.{0,20}(?:passed|resolved|stopped|went\s+away|brief|moment)|brief\s+(?:choking|suffocating)|couldn'?t\s+speak.{0,15}(?:second|moment|brief)",
     ["E_128"]),
    (r"не\s+мог\s+дышать.{0,15}(?:прошло|секунд|момент|кратко)|кратковременн\w+\s+(?:удушье|остановка\s+дыхани)",
     ["E_128"]),

    # ── Nocturnal dyspnea → Pulmonary edema / CHF ────────────────────────────
    (r"wake\s+up.{0,15}(?:breathless|can'?t\s+breathe|gasping|choking)|(?:breathless|choking|dyspnea).{0,10}(?:night|sleep|wak)",
     ["E_67"]),
    (r"просыпаюсь\s+(?:от\s+)?(?:удушь|нехватки\s+воздуха|задыха)|ночн\w+\s+(?:удушь|одышк|приступ)",
     ["E_67"]),

    # ── Swelling → Localized edema / CHF / Anaphylaxis ───────────────────────
    (r"swell\w*|edema|swollen\s+(?:leg|ankle|feet|face|lip|tongue)|puff\w+\s+(?:up|face|eyes)",
     ["E_151"]),
    (r"отёк|опухл\w+|распухл|отекл\w+\s+(?:ног|лиц|губ|язык)|припухлост",
     ["E_151"]),

    # ── Loss of appetite → Neoplasm / TB / Anemia ────────────────────────────
    (r"(?:lost|loss\s+of|no)\s+appetite|don'?t\s+(?:feel\s+like|want\s+to)\s+eat|not\s+hungry|eating\s+less",
     ["E_161"]),
    (r"нет\s+аппетит|пропал\s+аппетит|не\s+хочу\s+есть|потеря\s+аппетит",
     ["E_161"]),

    # ── Red cheeks / flushing → Scombroid food poisoning / SLE ────────────────
    (r"(?:cheeks?|face)\s+(?:turned|flush|red|burning)|flushing|face\s+is\s+red|red\s+(?:cheeks?|face)",
     ["E_92"]),
    (r"щёки\s+(?:красн|горят|покраснел)|лицо\s+(?:красн|покраснел|горит)|приливы\s+(?:к\s+)?лиц",
     ["E_92"]),

    # ── After eating fish → Scombroid ─────────────────────────────────────────
    (r"(?:ate|eat\w*)\s+(?:bad\s+)?(?:fish|tuna|mackerel|sushi)|fish\s+(?:poison|allergy)|after\s+(?:eat\w*\s+)?fish",
     ["E_42"]),  # contact with allergen (fish)
    (r"(?:съел|ел|после)\s+(?:рыб|тунц|суши)|рыб\w+\s+(?:отравлен|аллерги)|после\s+рыб",
     ["E_42"]),

    # ── Unusual bleeding / bruising → SLE / Anemia ───────────────────────────
    (r"bruis\w+\s+(?:easily|without\s+reason)|unexplained\s+(?:bruise|bleed)|unusual\s+(?:bleed|bruise)",
     ["E_178"]),
    (r"синяки\s+(?:без\s+причин|легко)|необъяснимые\s+(?:синяки|кровотечен)|кровоточ\w+\s+(?:без\s+причин|дёсн)",
     ["E_178"]),

    # ── Irregular heartbeat (chaotic) → Atrial fibrillation ───────────────────
    (r"irregular\w*\s+(?:heart|pulse|beat|rhythm)|(?:heart|pulse)\s+(?:all\s+over|chaotic|erratic|skipping|irregular)|missing\s+(?:a\s+)?beat|arrhythmi|beating.{0,10}irregular",
     ["E_164"]),
    (r"нерегулярн\w+\s+(?:пульс|сердцебиени|ритм)|аритми|перебои\s+в\s+(?:сердц|ритм)|сердце\s+(?:замирает|пропускает)",
     ["E_164"]),

    # ── Double vision → Myasthenia gravis ─────────────────────────────────────
    (r"double\s+vision|see\w*\s+(?:double|two)|diplopia|two\s+(?:images?|of\s+everything)",
     ["E_52"]),
    (r"двоится\s+в\s+глазах|двоение|дипло|вижу\s+(?:двойн|два\s+изображен)",
     ["E_52"]),

    # ── Mouth ulcers → SLE ────────────────────────────────────────────────────
    (r"mouth\s+(?:ulcer|sore)|(?:ulcer|sore)\s+(?:in\s+)?(?:my\s+)?mouth|canker\s+sore|oral\s+ulcer",
     ["E_206"]),
    (r"язв\w+\s+(?:во\s+)?рту|стоматит|ранки\s+во\s+рту|болячк\w+\s+во\s+рту",
     ["E_206"]),

    # ── Pale stool + dark urine → Hepatic / Pancreatic ──────────────────────
    (r"pale\s+stool|(?:light|white|clay)\s+(?:colored\s+)?stool|dark\s+urine|urine\s+(?:is\s+)?(?:dark|brown|tea)",
     ["E_188"]),
    (r"светл\w+\s+(?:кал|стул)|бесцветн\w+\s+(?:кал|стул)|тёмн\w+\s+моч|моча\s+(?:тёмн|коричнев)",
     ["E_188"]),

    # ── Breathless with minimal effort → Pulmonary edema / CHF / Anemia ──────
    (r"(?:breathless|winded|panting).{0,15}(?:minimal|slight|little|small)\s+(?:effort|exertion|activity)|out\s+of\s+breath.{0,10}(?:walk|stair|dress)",
     ["E_64"]),
    (r"одышка\s+при\s+(?:малейш|миним|небольш)\w+\s+(?:нагрузк|усили)|задыхаюсь\s+(?:при\s+ходьбе|на\s+лестнице)",
     ["E_64"]),

    # ── Extreme fatigue / bedridden → Severe conditions ──────────────────────
    (r"(?:so\s+tired|exhausted).{0,15}(?:can'?t\s+get\s+up|stay\s+in\s+bed|can'?t\s+function)|bedridden|stuck\s+in\s+bed",
     ["E_88"]),
    (r"(?:настолько|так)\s+(?:устал|слаб).{0,10}(?:не\s+могу\s+встать|лежу\s+весь)|лежу\s+(?:целый|весь)\s+день|прикован\s+к\s+кровати",
     ["E_88"]),

    # ── Confusion / disorientation → Severe infections / Neuro ───────────────
    (r"confus\w+|disorient\w+|(?:don'?t|can'?t)\s+(?:know|remember)\s+where\s+(?:i\s+)?am|mental\s+(?:fog|confusion)",
     ["E_39"]),
    (r"спутанност\w+\s+сознани|дезориентац|не\s+понимаю\s+где\s+(?:я|нахожусь)|заторможен",
     ["E_39"]),

    # ── Loss of consciousness → Syncope / Seizures ──────────────────────────
    (r"(?:lost|loss\s+of)\s+consciousness|passed\s+out|blacked\s+out|fainted",
     ["E_159"]),
    (r"потерял\s+сознани|потеря\s+сознани|упал\s+в\s+обморок|отключился",
     ["E_159"]),

    # ── Seizures / convulsions ────────────────────────────────────────────────
    (r"seizure|convulsion|(?:violent|sustained)\s+muscle\s+contraction|epilep\w*\s+(?:fit|attack)|absense\s+episode",
     ["E_43"]),
    (r"судорог\w+|конвульси|припадок|эпилепт\w+\s+(?:приступ|припадок)|потеря\s+сознания\s+с\s+судорог",
     ["E_43"]),

    # ── Side chest pain → Pleurisy / Pneumothorax / PE ───────────────────────
    (r"pain\s+(?:in\s+)?(?:my\s+)?(?:left|right)\s+(?:side\s+(?:of\s+)?)?chest|(?:left|right)\s+(?:side\s+)?chest\s+pain|(?:pain|stab)\w*\s+(?:in\s+)?(?:my\s+)?(?:left|right)\s+(?:lung|rib)",
     ["E_53", "E_55_@_V_55"]),  # side of chest(R)
    (r"боль\s+(?:в\s+)?(?:лев|прав)\w+\s+(?:стороне\s+)?(?:груд|бок)|колет\s+(?:в\s+)?(?:лев|прав)\w+\s+(?:бок|груд)",
     ["E_53", "E_55_@_V_55"]),

    # ── Chest wall pain / rib pain → Spontaneous rib fracture ────────────────
    (r"rib\s+(?:pain|hurts|fractur|crack)|pain\w*\s+(?:in\s+)?(?:my\s+)?rib|(?:crack|broke)\w*\s+(?:a\s+)?rib",
     ["E_53", "E_55_@_V_55", "E_216"]),  # chest + worse with movement
    (r"(?:перелом|трещин)\w+\s+ребр|ребро\s+(?:болит|сломал)|боль.{0,10}ребр",
     ["E_53", "E_55_@_V_55", "E_216"]),

    # ── Barking cough → Croup ─────────────────────────────────────────────────
    (r"barking\s+cough|croup|seal.{0,5}(?:like\s+)?cough|stridor\w*\s+(?:child|baby|infant|kid)",
     ["E_201", "E_112"]),
    (r"лающий\s+каш\w+|круп|каш\w+\s+(?:как\s+)?лай",
     ["E_201", "E_112"]),
]

# ─── BM25 synonym expansion (for fallback BM25 only) ─────────────────────────
_BM25_SYNONYMS: Dict[str, List[str]] = {
    "температура": ["fever", "temperature"],
    "жар": ["fever", "temperature"],
    "озноб": ["chills", "shivers"],
    "ломота": ["muscle pain", "body aches"],
    "насморк": ["runny nose", "nasal discharge"],
    "кашель": ["cough"],
    "мокрота": ["sputum", "phlegm"],
    "одышка": ["shortness breath"],
    "тошнота": ["nausea"],
    "рвота": ["vomit"],
    "боль в груди": ["chest pain"],
    "потею": ["sweating"],
    "головокружение": ["dizziness"],
    "горло болит": ["sore throat"],
    "боль в животе": ["abdominal pain"],
    "сыпь": ["rash"],
    "зуд": ["itching"],
    "сердцебиение": ["palpitations"],
    "слабость": ["fatigue", "weakness"],
    "усталость": ["fatigue"],
    "головная боль": ["headache", "cephalalgia"],
    "мигрень": ["migraine", "headache"],
    "болит голова": ["headache"],
    "пульсирующая": ["throbbing", "pulsating"],
    "светобоязнь": ["photophobia", "light sensitivity"],
    "фотофобия": ["photophobia"],
    "фонофобия": ["phonophobia"],
    "аура": ["aura", "visual disturbance"],
    "висок": ["temple", "temporal"],
    "затылок": ["occiput", "occipital"],
    # Kazakh → English expansions for BM25
    "жөтел": ["cough"],
    "қызба": ["fever", "temperature"],
    "бас ауру": ["headache"],
    "жүрек айну": ["nausea"],
    "құсу": ["vomit"],
    "іш ауру": ["abdominal pain", "stomach"],
    "ентігу": ["shortness breath", "dyspnea"],
    "тамақ ауру": ["sore throat"],
    "мұрын": ["nasal", "nose"],
    "бөртпе": ["rash"],
    "қышыма": ["itching"],
    "әлсіздік": ["fatigue", "weakness"],
    "бас айналу": ["dizziness"],
    "кеуде ауру": ["chest pain"],
    "терлеу": ["sweating"],
}


def _tokenize(text: str) -> List[str]:
    """Simple multilingual tokenizer: lowercase, split on non-alphanumeric."""
    text = text.lower()
    # а-яё = Russian, әғқңөұүһі = Kazakh-specific Cyrillic
    tokens = re.findall(r"[a-zа-яёәғқңөұүһіüöäéàèùâêîôûç]+", text)
    return tokens


def _expand_query(text: str) -> str:
    """Expand Russian patient text with English medical synonyms for BM25."""
    text_lower = text.lower()
    extra = []
    for ru_term, expansions in _BM25_SYNONYMS.items():
        if ru_term in text_lower:
            extra.extend(expansions)
    return text + " " + " ".join(extra) if extra else text


def _extract_age_sex(text: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract age and sex from Russian/English patient text using regex."""
    age: Optional[int] = None
    sex: Optional[str] = None

    age_patterns = [
        r"(\d+)\s*лет\b",
        r"(\d+)\s*год[а]?\b",
        r"мне\s+(\d+)",
        r"возраст\s*[:\-]?\s*(\d+)",
        r"(\d+)\s*жаст?\w*",          # KK: 25 жаста / жасымда
        r"менің\s+жасым\s+(\d+)",      # KK: менің жасым 25
        r"маған\s+(\d+)",              # KK: маған 25
        r"age\s*[:\-]?\s*(\d+)",
        r"(\d+)\s*years?\s*old",
        r"(\d+)[- ]year[- ]old",
    ]
    for pat in age_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            candidate = int(m.group(1))
            if 1 <= candidate <= 120:
                age = candidate
                break

    female_re = r"женщина|женщине|женского|девушка|әйел|қыз|woman|female|girl"
    male_re = r"мужчина|мужчине|мужского|парень|мальчик|ер\s*адам|ұл|(?<!\w)man(?!\w)|male|boy"

    if re.search(female_re, text, re.IGNORECASE):
        sex = "F"
    elif re.search(male_re, text, re.IGNORECASE):
        sex = "M"

    return age, sex


_NEGATION_BEFORE_RE = re.compile(
    r"(?:^|[\s,;.!?])(?:не\s+|нет\s+|без\s+|отсутств\w+\s+|никак\w+\s+|ни\s+|жоқ\s+|емес\s+)",
    re.MULTILINE,
)
_NEGATION_AFTER_RE = re.compile(
    r"^\w*\s+(?:нет|отсутствует|не\s+было|не\s+беспокоит|жоқ|емес)\b"
)


def _is_negated(text: str, match_start: int, match_end: int = 0) -> bool:
    """
    Check if the match is negated — looks both backward and forward.
    Backward: "не кашляю", "без температуры"
    Forward:  "кашля нет", "жөтел жоқ"
    """
    # Backward check (up to 15 chars before match)
    window_start = max(0, match_start - 15)
    window_before = text[window_start:match_start]
    if _NEGATION_BEFORE_RE.search(window_before):
        return True

    # Forward check (up to 25 chars after match end)
    if match_end > 0:
        window_after = text[match_end:match_end + 25]
        if _NEGATION_AFTER_RE.search(window_after):
            return True

    return False


def _vocab_extract(text: str) -> List[str]:
    """
    Direct vocabulary lookup: scan patient text for symptom terms,
    return the corresponding DDXPlus evidence IDs.
    High precision — only returns IDs when pattern clearly matches.
    Uses finditer to check ALL occurrences: if at least one is not negated,
    the symptom is present.  E.g. "кашля нет, но вчера кашлял" → cough present.
    """
    text_lower = text.lower()
    matched: Set[str] = set()
    _flags = re.DOTALL | re.MULTILINE

    for pattern, ev_ids in _VOCAB:
        for m in re.finditer(pattern, text_lower, _flags):
            if not _is_negated(text_lower, m.start(), m.end()):
                matched.update(ev_ids)
                break  # one non-negated hit is enough for this pattern

    return list(matched)


# ─── BM25 retriever (fallback for unknown symptoms) ──────────────────────────

class EvidenceRetriever:
    """
    BM25-based retrieval over DDXPlus evidence questions.
    Used as FALLBACK when vocabulary extraction returns few evidences.
    """

    def __init__(self, evidences_path: Path = _EVIDENCES_PATH):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 not installed. Run: pip install rank-bm25")

        with open(evidences_path, encoding="utf-8") as f:
            self._meta: Dict = json.load(f)

        from ml_model import _QUESTION_TRANSLATIONS
        self._ids: List[str] = []
        corpus_tokens: List[List[str]] = []

        for ev_id, meta in self._meta.items():
            q_en = meta.get("question_en", "")
            ru_trans = _QUESTION_TRANSLATIONS.get(ev_id, {})
            q_ru = ru_trans.get("ru", "") if isinstance(ru_trans, dict) else ""
            q_kk = ru_trans.get("kk", "") if isinstance(ru_trans, dict) else ""

            combined = f"{q_en} {q_ru} {q_kk}"
            tokens = _tokenize(combined)
            if tokens:
                self._ids.append(ev_id)
                corpus_tokens.append(tokens)

        self._bm25 = BM25Okapi(corpus_tokens)
        logger.info("EvidenceRetriever: indexed %d evidences", len(self._ids))

    def retrieve(self, text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Retrieve top_k most relevant evidence IDs."""
        expanded = _expand_query(text)
        query_tokens = _tokenize(expanded)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        results = sorted(
            zip(self._ids, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(ev_id, score) for ev_id, score in results[:top_k] if score > 0]

    def extract_direct(
        self,
        text: str,
        top_k: int = 25,
        bm25_fallback_threshold: float = 8.0,
        bm25_fallback_max: int = 5,
    ) -> Tuple[List[str], Optional[int], Optional[str]]:
        """
        Main extraction method — vocabulary first, BM25 fallback for unknowns.

        Steps:
          1. Vocabulary lookup: precise Russian→E_XX direct mapping.
          2. If vocabulary returns <4 evidences, add BM25 top candidates
             (only high-confidence: score > 8.0) to fill gaps.
          3. Extract age and sex via regex.

        Returns: (evidences: List[str], age: Optional[int], sex: Optional[str])
        """
        age, sex = _extract_age_sex(text)

        # Step 1: Vocabulary lookup (high precision)
        vocab_evs = _vocab_extract(text)
        matched_set: Set[str] = set(vocab_evs)

        logger.info(
            "vocab_extract: %r -> %d evidences: %s",
            text[:60], len(vocab_evs), vocab_evs,
        )

        # Step 2: BM25 fallback — only if vocabulary found few evidences
        if len(matched_set) < 4:
            try:
                candidates = self.retrieve(text, top_k=top_k)
                if candidates:
                    logger.info(
                        "BM25 scores (top-10): %s",
                        [(eid, round(s, 2)) for eid, s in candidates[:10]],
                    )
                added = 0
                for ev_id, score in candidates:
                    if score < bm25_fallback_threshold:
                        break
                    meta = self._meta.get(ev_id, {})
                    # Skip: categorical, antecedent (history), already matched
                    if meta.get("data_type", "B") != "B":
                        continue
                    if meta.get("is_antecedent", False):
                        continue
                    if ev_id in matched_set:
                        continue
                    matched_set.add(ev_id)
                    added += 1
                    if added >= bm25_fallback_max:
                        break
                if added:
                    logger.info("BM25 fallback added %d high-confidence evidences", added)
            except Exception as e:
                logger.warning("BM25 fallback failed: %s", e)

        evidences = list(matched_set)
        logger.info(
            "extract_direct result: %r -> %d evidences (age=%s sex=%s): %s",
            text[:60], len(evidences), age, sex, evidences,
        )
        return evidences, age, sex


# ─── Singleton ────────────────────────────────────────────────────────────────
_retriever: EvidenceRetriever | None = None


def get_retriever() -> EvidenceRetriever:
    global _retriever
    if _retriever is None:
        _retriever = EvidenceRetriever()
    return _retriever
