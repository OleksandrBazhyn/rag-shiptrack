"""
Демонстраційний прототип RAG (Retrieval Augmented Generation).

Консольний чат, який відповідає на запитання про інформаційну систему
ShipTrack на основі документації з каталогу knowledge_base.

Етапи роботи:
  1. Індексування: документи діляться на фрагменти, для кожного фрагмента
     обчислюється векторне представлення (embedding), індекс кешується у файл.
  2. Пошук: запитання користувача перетворюється на вектор, за косинусною
     подібністю відбираються k найближчих фрагментів.
  3. Генерація: знайдені фрагменти разом із запитанням передаються LLM,
     яка формує відповідь лише на основі наданого контексту.

Запуск:
  python rag_chat.py          інтерактивний чат
  python rag_chat.py --demo   прогін тестових запитань з RAG і без RAG

Автор: Бажин Олександр Вадимович, Остапенко Дмитро Миколайович, 2026.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
INDEX_PATH = BASE_DIR / "index.json"
DEMO_RESULTS_PATH = BASE_DIR / "demo_results.md"

load_dotenv(BASE_DIR / ".env")

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
CHUNK_SIZE = 800      # максимальна довжина тексту фрагмента без заголовка, символів
CHUNK_OVERLAP = 1     # скільки останніх рядків переносити в наступний фрагмент

# Ціни за 1 млн токенів у доларах США (для оцінки витрат, змінюються в .env)
PRICE_IN = float(os.getenv("PRICE_INPUT_PER_1M", "0.15"))
PRICE_OUT = float(os.getenv("PRICE_OUTPUT_PER_1M", "0.60"))
PRICE_EMBED = float(os.getenv("PRICE_EMBED_PER_1M", "0.02"))

SYSTEM_PROMPT_RAG = (
    "Ти асистент проєктної команди інформаційної системи ShipTrack "
    "(кур'єрська служба доставки «останньої милі»). "
    "Відповідай українською мовою лише на основі наданого контексту. "
    "Якщо в контексті немає відповіді, прямо скажи, що в документації "
    "ShipTrack такої інформації немає, і нічого не вигадуй. "
    "Після кожного твердження вказуй номер фрагмента у квадратних дужках, "
    "наприклад [2]. Відповідай стисло, до 5 речень."
)

SYSTEM_PROMPT_PLAIN = (
    "Ти асистент. Відповідай українською мовою стисло, до 5 речень."
)

DEMO_QUESTIONS = [
    "О котрій годині мають бути затверджені маршрути кур'єрів і коли кур'єри виїжджають?",
    "Яку частку доставок з першої спроби має забезпечити ShipTrack і яким є поточний показник?",
    "Що відбувається з посилкою після другої невдалої спроби доставки?",
    "Як у ShipTrack захищено номер телефону одержувача?",
    "Чому сервіс маршрутизації виділено окремо і на яких технологіях він побудований?",
    "Яка заробітна плата кур'єра в ShipTrack?",
]


# індексування

def load_documents(kb_dir: Path) -> list[dict]:
    """Зчитує всі файли .md і .txt з каталогу бази знань."""
    docs = []
    for path in sorted(kb_dir.glob("*")):
        if path.suffix.lower() in (".md", ".txt"):
            docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs


def split_into_chunks(doc: dict) -> list[dict]:
    """
    Ділить документ на фрагменти довжиною до CHUNK_SIZE символів.
    Межі фрагментів проходять по рядках, тому речення і пункти списку
    не розриваються. До кожного фрагмента додається назва документа та
    розділу, що покращує якість пошуку.
    """
    title, section = "", ""
    chunks, current = [], []

    def flush():
        if current:
            header = f"{title}. {section}".strip(". ")
            chunks.append({
                "source": doc["source"],
                "text": header + "\n" + "\n".join(current),
            })

    for line in doc["text"].splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            title = line[2:]
            continue
        if line.startswith("## "):
            flush()
            current = []
            section = line[3:]
            continue
        if current and len("\n".join(current + [line])) > CHUNK_SIZE:
            flush()
            current = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else []
        current.append(line)
    flush()
    return chunks


def embed(client: OpenAI, texts: list[str]) -> tuple[np.ndarray, int]:
    """Повертає нормовані вектори для списку текстів і кількість токенів."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors, response.usage.total_tokens


def kb_fingerprint(docs: list[dict]) -> str:
    """Хеш бази знань і параметрів: якщо щось змінилося, індекс будується заново."""
    h = hashlib.sha256(f"{EMBED_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    for d in docs:
        h.update(d["source"].encode())
        h.update(d["text"].encode())
    return h.hexdigest()


def build_or_load_index(client: OpenAI, force: bool = False) -> dict:
    docs = load_documents(KB_DIR)
    if not docs:
        sys.exit(f"У каталозі {KB_DIR} немає документів .md або .txt")
    fingerprint = kb_fingerprint(docs)

    if INDEX_PATH.exists() and not force:
        cached = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            cached["vectors"] = np.array(cached["vectors"], dtype=np.float32)
            print(f"Індекс завантажено з кешу: {len(cached['chunks'])} фрагментів.")
            return cached

    chunks = [c for d in docs for c in split_into_chunks(d)]
    print(f"Побудова індексу: {len(docs)} документів, {len(chunks)} фрагментів...")
    vectors, tokens = embed(client, [c["text"] for c in chunks])
    index = {"fingerprint": fingerprint, "model": EMBED_MODEL, "chunks": chunks,
             "vectors": vectors.tolist()}
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"Індекс збережено ({tokens} токенів, ~${tokens * PRICE_EMBED / 1e6:.5f}).")
    index["vectors"] = vectors
    return index


# пошук

def retrieve(client: OpenAI, index: dict, question: str, k: int) -> tuple[list[dict], int]:
    """Повертає k найбільш релевантних фрагментів з оцінкою подібності."""
    query_vec, tokens = embed(client, [question])
    scores = index["vectors"] @ query_vec[0]          # косинусна подібність
    best = np.argsort(scores)[::-1][:k]
    found = [dict(index["chunks"][i], score=float(scores[i])) for i in best]
    return found, tokens


# генерація

def build_messages(question: str, fragments: list[dict] | None) -> list[dict]:
    if fragments is None:
        return [{"role": "system", "content": SYSTEM_PROMPT_PLAIN},
                {"role": "user", "content": question}]
    context = "\n\n".join(
        f"[{i}] (джерело: {f['source']})\n{f['text']}" for i, f in enumerate(fragments, 1)
    )
    user = f"Контекст:\n{context}\n\nЗапитання: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT_RAG},
            {"role": "user", "content": user}]


class Assistant:
    def __init__(self, client: OpenAI, index: dict):
        self.client = client
        self.index = index
        self.k = TOP_K
        self.last_fragments: list[dict] = []
        self.cost = 0.0
        self.last_usage = (0, 0)

    def ask(self, question: str, use_rag: bool = True) -> str:
        fragments = None
        if use_rag:
            fragments, emb_tokens = retrieve(self.client, self.index, question, self.k)
            self.last_fragments = fragments
            self.cost += emb_tokens * PRICE_EMBED / 1e6

        messages = build_messages(question, fragments)
        try:
            response = self.client.chat.completions.create(
                model=CHAT_MODEL, messages=messages, temperature=0.2)
        except BadRequestError as exc:
            # моделі з міркуваннями (наприклад, gpt-5-mini) не приймають temperature
            if "temperature" not in str(exc):
                raise
            response = self.client.chat.completions.create(model=CHAT_MODEL, messages=messages)
        usage = response.usage
        self.cost += (usage.prompt_tokens * PRICE_IN + usage.completion_tokens * PRICE_OUT) / 1e6
        answer = response.choices[0].message.content.strip()
        self.last_usage = (usage.prompt_tokens, usage.completion_tokens)
        return answer


# консоль

HELP = """Команди:
  <запитання>          відповідь з використанням RAG
  /norag <запитання>   відповідь без контексту (лише знання моделі)
  /compare <запитання> обидві відповіді для порівняння
  /sources             фрагменти, знайдені для останнього запитання
  /k <число>           кількість фрагментів контексту (зараз {k})
  /reindex             перебудувати індекс
  /help                ця довідка
  /exit                вихід"""


def print_sources(fragments: list[dict]) -> None:
    if not fragments:
        print("Ще не було запитань з RAG.")
        return
    for i, f in enumerate(fragments, 1):
        preview = f["text"].replace("\n", " ")[:150]
        print(f"  [{i}] {f['source']}  подібність={f['score']:.3f}\n      {preview}...")


def print_answer(label: str, assistant: Assistant, answer: str) -> None:
    p, c = assistant.last_usage
    print(f"\n{label}:\n{answer}\n  (токени: запит {p}, відповідь {c})")


def interactive(assistant: Assistant) -> None:
    print(f"\nRAG-асистент ShipTrack. Модель: {CHAT_MODEL}, embeddings: {EMBED_MODEL}.")
    print(HELP.format(k=assistant.k))
    while True:
        try:
            line = input("\nВи> ").strip()
        except (EOFError, KeyboardInterrupt):
            line = "/exit"
        if not line:
            continue
        cmd, _, arg = line.partition(" ")
        try:
            if cmd == "/exit":
                print(f"Орієнтовні витрати сесії: ${assistant.cost:.5f}")
                break
            elif cmd == "/help":
                print(HELP.format(k=assistant.k))
            elif cmd == "/sources":
                print_sources(assistant.last_fragments)
            elif cmd == "/k":
                assistant.k = max(1, int(arg))
                print(f"k = {assistant.k}")
            elif cmd == "/reindex":
                assistant.index = build_or_load_index(assistant.client, force=True)
            elif cmd in ("/norag", "/compare") and not arg.strip():
                print(f"Вкажіть запитання після команди, наприклад: {cmd} Коли виїжджають кур'єри?")
            elif cmd == "/norag":
                print_answer("Без RAG", assistant, assistant.ask(arg, use_rag=False))
            elif cmd == "/compare":
                print_answer("Без RAG", assistant, assistant.ask(arg, use_rag=False))
                print_answer("З RAG", assistant, assistant.ask(arg, use_rag=True))
                print_sources(assistant.last_fragments)
            elif cmd.startswith("/"):
                print("Невідома команда. Введіть /help.")
            else:
                print_answer("Асистент", assistant, assistant.ask(line))
                print("  Джерела: " + ", ".join(
                    f"[{i}] {f['source']}" for i, f in enumerate(assistant.last_fragments, 1)))
        except Exception as exc:  # помилки мережі, ключа API тощо
            print(f"Помилка: {exc}")


def run_demo(assistant: Assistant) -> None:
    """Проганяє тестові запитання в двох режимах і зберігає звіт у Markdown."""
    lines = ["# Результати демонстраційного прогону",
             f"Дата: {datetime.now():%Y-%m-%d %H:%M}, модель: {CHAT_MODEL}, "
             f"embeddings: {EMBED_MODEL}, k = {assistant.k}\n"]
    for n, q in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n=== Запитання {n}: {q}")
        plain = assistant.ask(q, use_rag=False)
        print_answer("Без RAG", assistant, plain)
        rag = assistant.ask(q, use_rag=True)
        print_answer("З RAG", assistant, rag)
        print_sources(assistant.last_fragments)
        sources = "; ".join(f"[{i}] {f['source']} ({f['score']:.3f})"
                            for i, f in enumerate(assistant.last_fragments, 1))
        lines += [f"## Запитання {n}", q, "", "**Без RAG:**", plain, "",
                  "**З RAG:**", rag, "", f"Знайдені фрагменти: {sources}", ""]
    lines.append(f"Орієнтовні витрати прогону: ${assistant.cost:.5f}")
    DEMO_RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nРезультати збережено у {DEMO_RESULTS_PATH.name}. Витрати: ${assistant.cost:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-асистент для документації ShipTrack")
    parser.add_argument("--demo", action="store_true", help="прогнати тестові запитання")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Не задано OPENAI_API_KEY. Скопіюйте .env.example у .env і вкажіть ключ.")

    client = OpenAI()
    assistant = Assistant(client, build_or_load_index(client))
    run_demo(assistant) if args.demo else interactive(assistant)


if __name__ == "__main__":
    main()
