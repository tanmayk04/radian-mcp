import os
from dotenv import load_dotenv
import mysql.connector

load_dotenv()

cnx = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST", "127.0.0.1"),
    port=int(os.getenv("MYSQL_PORT", "3306")),
    user=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", ""),
    database=os.getenv("MYSQL_DATABASE", "radianreporting"),
)

cur = cnx.cursor()
cur.execute("SELECT DATABASE(), COUNT(*) FROM information_schema.tables WHERE table_schema = DATABASE();")
print(cur.fetchone())
cur.close()
cnx.close()
