"""inkprint app package.

Side effect: loads .env once at first import so every submodule that reads
os.getenv() — llm, search, ... — sees the same configuration regardless of
which one happens to be imported first.
"""

from dotenv import load_dotenv

load_dotenv()
