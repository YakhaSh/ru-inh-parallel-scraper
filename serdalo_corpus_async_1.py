# ================================================
# Асинхронный сбор параллельных текстов (ингушский ↔ русский)
# Автор: Шанхоева Яха Ахмет-Башировна
# Цель: найти одинаковые статьи на сайте serdalo.ru на двух языках
# и сохранить их как пары текстов для корпуса машинного перевода.
# ================================================

# ---------- ИМПОРТЫ ----------
import asyncio        # асинхронность — обработка множества страниц одновременно
import aiohttp         # библиотека для асинхронных HTTP-запросов
from bs4 import BeautifulSoup  # для извлечения текста из HTML
import pandas as pd    # для работы с таблицами (TSV)
import re, os, json, time
from tqdm.asyncio import tqdm  # прогресс-бар для асинхронных операций
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF анализ текста
from sklearn.metrics.pairwise import cosine_similarity        # измерение похожести
import argparse
import logging          # для ведения лога (журнала событий)

# ---------- НАСТРОЙКИ ----------
BASE_URL = "https://serdalo.ru"                      # адрес сайта
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"              # путь к карте сайта
SAVE_DIR = "data/parallel"                           # папка для сохранения результатов
TSV_FILE = os.path.join(SAVE_DIR, "parallel_ru_inh.tsv")
JSONL_FILE = os.path.join(SAVE_DIR, "parallel_ru_inh.jsonl")

SIM_THRESHOLD = 0.18      # минимальное значение похожести текстов (0–1)
CONCURRENCY = 15          # сколько страниц обрабатываем одновременно

# создаём папку для сохранения, если нет
os.makedirs(SAVE_DIR, exist_ok=True)

# настройка логирования (журнал программы)
logging.basicConfig(
    filename=os.path.join(SAVE_DIR, "parser.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------- ПРОВЕРКА ЯЗЫКА ----------
# Эвристики для распознавания ингушского и русского языков
INH_PATTERNS = ["ӏ", "Ӏ", "гӏ", "кӏ", "кх", "хь", "къ", "хъ", "ӏа", "ӏе", "ӏо", "ӏу", "ӏи"]
RU_COMMON = set("и в не на что с я он она они был была были это как по от из у за".split())

def detect_inh_ratio(text):
    """Подсчёт доли ингушских букв"""
    text = text.lower()
    total = len(text)
    count = sum(len(re.findall(pat, text)) for pat in INH_PATTERNS)
    return count / max(total, 1)

def ru_ratio(text):
    """Подсчёт доли частых русских слов"""
    words = re.findall(r"[а-яё]+", text.lower())
    if not words:
        return 0
    known = sum(w in RU_COMMON for w in words)
    return known / len(words)

def detect_lang(text):
    """Определение языка текста"""
    inh_r = detect_inh_ratio(text)
    ru_r = ru_ratio(text)
    if inh_r > 0.008 and ru_r < 0.6:
        return "inh"      # ингушский
    elif inh_r < 0.002 and ru_r > 0.5:
        return "ru"       # русский
    else:
        return "mixed"    # смешанный


# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------

async def fetch(session, url):
    """Асинхронно загружает HTML-страницу"""
    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status == 200:
                return await resp.text()
    except Exception as e:
        logging.warning(f"Ошибка загрузки {url}: {e}")
    return None


def extract_article(html):
    """Извлекает заголовок и текст статьи"""
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find(["h1", "h2"])
    title = title_tag.get_text(strip=True) if title_tag else ""

    content = (
        soup.find("article")
        or soup.find("div", class_=re.compile("post|content|entry"))
        or soup
    )

    paragraphs = content.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text.split()) < 30:
        return None, None
    return title, text


def hybrid_similarity(a, b):
    """Сравнение двух текстов по словам и буквенным цепочкам"""
    if not a or not b:
        return 0

    vect_word = TfidfVectorizer(analyzer="word", min_df=2).fit([a, b])
    sim_word = cosine_similarity(vect_word.transform([a, b]))[0][1]

    vect_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit([a, b])
    sim_char = cosine_similarity(vect_char.transform([a, b]))[0][1]

    # символы важнее слов: 70% символы + 30% слова
    return 0.3 * sim_word + 0.7 * sim_char


def find_russian_alternate(url_inh, html):
    """Ищет русскую версию статьи"""
    soup = BeautifulSoup(html, "html.parser")

    link = soup.find("a", string=re.compile("Русский", re.IGNORECASE))
    if link and link.get("href"):
        href = link["href"]
        return href if href.startswith("http") else BASE_URL.rstrip("/") + href

    # если ссылки нет — пробуем заменить /inh/ на /
    if "/inh/" in url_inh:
        return url_inh.replace("/inh/", "/")
    return None


async def load_sitemap():
    """Загружает sitemap.xml и выбирает ингушские страницы"""
    async with aiohttp.ClientSession() as session:
        xml = await fetch(session, SITEMAP_URL)
        if not xml:
            print("Не удалось загрузить sitemap.xml")
            return []
        soup = BeautifulSoup(xml, "xml")
        urls = [loc.get_text() for loc in soup.find_all("loc")]
        urls = [u for u in urls if "/inh/" in u and all(x not in u for x in ["/tag/", "/page/", "/category/"])]
        return urls


# ---------- ОСНОВНАЯ ОБРАБОТКА ОДНОЙ СТРАНИЦЫ ----------

async def process_page(session, url_inh, results):
    """Обрабатывает одну ингушскую статью и находит к ней русскую"""
    html_inh = await fetch(session, url_inh)
    if not html_inh:
        return

    url_ru = find_russian_alternate(url_inh, html_inh)
    if not url_ru:
        return

    html_ru = await fetch(session, url_ru)
    if not html_ru:
        return

    title_inh, text_inh = extract_article(html_inh)
    title_ru, text_ru = extract_article(html_ru)
    if not text_inh or not text_ru:
        return

    lang_inh = detect_lang(text_inh)
    lang_ru = detect_lang(text_ru)
    if lang_inh not in ("inh", "mixed") or lang_ru not in ("ru", "mixed"):
        return

    sim = hybrid_similarity(text_ru, text_inh)
    if sim < SIM_THRESHOLD:
        return

    # добавляем результат в общий список
    entry = {
        "url_ru": url_ru,
        "url_inh": url_inh,
        "title_ru": title_ru,
        "title_inh": title_inh,
        "text_ru": text_ru,
        "text_inh": text_inh,
        "similarity": round(sim, 3),
    }
    results.append(entry)
    logging.info(f"Добавлена пара: {url_ru} — {url_inh}, sim={sim:.3f}")


# ---------- ГЛАВНАЯ ФУНКЦИЯ ----------

async def build_parallel_pairs(limit=None):
    """Главная функция, собирающая все пары текстов"""
    urls_inh = await load_sitemap()
    if limit:
        urls_inh = urls_inh[:limit]
    print(f"Найдено {len(urls_inh)} ингушских страниц")

    results = []
    sem = asyncio.Semaphore(CONCURRENCY)
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        async def bounded(url):
            async with sem:
                await process_page(session, url, results)
                # автосейв каждые 50 найденных пар
                if len(results) % 50 == 0 and results:
                    df = pd.DataFrame(results)
                    df.to_csv(TSV_FILE, sep="\t", index=False)
                    with open(JSONL_FILE, "w", encoding="utf-8") as f:
                        for r in results:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    logging.info(f"Автосейв {len(results)} пар")

        # запускаем все задачи параллельно
        await tqdm.gather(*[bounded(url) for url in urls_inh])

    if not results:
        print("Пары не найдены")
        return

    df = pd.DataFrame(results)
    df.to_csv(TSV_FILE, sep="\t", index=False)
    with open(JSONL_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print(f"\nСохранено {len(df)} пар в {SAVE_DIR}")
    print(f"⏱ Время выполнения: {elapsed/60:.1f} мин")


# ---------- ТОЧКА ВХОДА ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Асинхронный сбор параллельных текстов serdalo.ru")
    parser.add_argument("--build", action="store_true", help="Построить пары (ингушский → русский)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить количество страниц")
    args = parser.parse_args()

    if args.build:
        asyncio.run(build_parallel_pairs(limit=args.limit))
    else:
        print("Укажите режим: --build")
