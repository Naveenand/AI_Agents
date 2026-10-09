import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
load_dotenv()

if 'port' not in os.environ:
    os.environ['port'] = '5432'

# ============================================================
# CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": os.environ['host'],
    "port": int(os.environ['port']),
    "database": os.environ['database'],
    "user": os.environ['user'],
    "password": os.environ['password'],
}

CSV_DIR = "data"


# ============================================================
# DATABASE CONNECTION
# ============================================================

conn = psycopg2.connect(**DB_CONFIG)
conn.autocommit = False

cursor = conn.cursor()

print("Connected to PostgreSQL")


# ============================================================
# CREATE TABLES
# ============================================================

create_tables_sql = """

CREATE SCHEMA IF NOT EXISTS public;

-- =========================================================
-- PRODUCT CATEGORY NAME TRANSLATION
-- =========================================================

CREATE TABLE IF NOT EXISTS public.product_category_name_translation (
    product_category_name VARCHAR(100) PRIMARY KEY,
    product_category_name_english VARCHAR(100) NOT NULL
);


-- =========================================================
-- CUSTOMERS
-- =========================================================

CREATE TABLE IF NOT EXISTS public.customers (
    customer_id CHAR(32) PRIMARY KEY,
    customer_unique_id CHAR(32) NOT NULL,
    customer_zip_code_prefix VARCHAR(5),
    customer_city VARCHAR(100),
    customer_state CHAR(2)
);


-- =========================================================
-- SELLERS
-- =========================================================

CREATE TABLE IF NOT EXISTS public.sellers (
    seller_id CHAR(32) PRIMARY KEY,
    seller_zip_code_prefix VARCHAR(5),
    seller_city VARCHAR(100),
    seller_state CHAR(2)
);


-- =========================================================
-- PRODUCTS
-- =========================================================
-- No FK to product_category_name_translation: two categories
-- in the products file (pc_gamer and
-- portateis_cozinha_e_preparadores_de_alimentos) have no
-- translation row, so an FK would reject the load.
-- Column names keep the dataset's original "lenght" spelling.

CREATE TABLE IF NOT EXISTS public.products (
    product_id CHAR(32) PRIMARY KEY,
    product_category_name VARCHAR(100),
    product_name_lenght INTEGER,
    product_description_lenght INTEGER,
    product_photos_qty INTEGER,
    product_weight_g INTEGER,
    product_length_cm INTEGER,
    product_height_cm INTEGER,
    product_width_cm INTEGER
);


-- =========================================================
-- ORDERS
-- =========================================================

CREATE TABLE IF NOT EXISTS public.orders (
    order_id CHAR(32) PRIMARY KEY,
    customer_id CHAR(32) NOT NULL,

    order_status VARCHAR(20),

    order_purchase_timestamp TIMESTAMP,
    order_approved_at TIMESTAMP,
    order_delivered_carrier_date TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP,

    CONSTRAINT fk_order_customer
        FOREIGN KEY (customer_id)
        REFERENCES public.customers(customer_id)
);


-- =========================================================
-- ORDER ITEMS
-- =========================================================

CREATE TABLE IF NOT EXISTS public.order_items (
    order_id CHAR(32) NOT NULL,
    order_item_id INTEGER NOT NULL,

    product_id CHAR(32) NOT NULL,
    seller_id CHAR(32) NOT NULL,

    shipping_limit_date TIMESTAMP,
    price DECIMAL(10,2),
    freight_value DECIMAL(10,2),

    CONSTRAINT pk_order_items
        PRIMARY KEY (order_id, order_item_id),

    CONSTRAINT fk_item_order
        FOREIGN KEY (order_id)
        REFERENCES public.orders(order_id),

    CONSTRAINT fk_item_product
        FOREIGN KEY (product_id)
        REFERENCES public.products(product_id),

    CONSTRAINT fk_item_seller
        FOREIGN KEY (seller_id)
        REFERENCES public.sellers(seller_id)
);


-- =========================================================
-- ORDER PAYMENTS
-- =========================================================

CREATE TABLE IF NOT EXISTS public.order_payments (
    order_id CHAR(32) NOT NULL,
    payment_sequential INTEGER NOT NULL,

    payment_type VARCHAR(20),
    payment_installments INTEGER,
    payment_value DECIMAL(10,2),

    CONSTRAINT pk_order_payments
        PRIMARY KEY (order_id, payment_sequential),

    CONSTRAINT fk_payment_order
        FOREIGN KEY (order_id)
        REFERENCES public.orders(order_id)
);


-- =========================================================
-- ORDER REVIEWS
-- =========================================================
-- review_id alone is not unique in this dataset (the same
-- review can cover several orders), so the key is
-- (review_id, order_id).

CREATE TABLE IF NOT EXISTS public.order_reviews (
    review_id CHAR(32) NOT NULL,
    order_id CHAR(32) NOT NULL,

    review_score INTEGER,
    review_comment_title VARCHAR(100),
    review_comment_message TEXT,

    review_creation_date TIMESTAMP,
    review_answer_timestamp TIMESTAMP,

    CONSTRAINT pk_order_reviews
        PRIMARY KEY (review_id, order_id),

    CONSTRAINT fk_review_order
        FOREIGN KEY (order_id)
        REFERENCES public.orders(order_id),

    CONSTRAINT chk_review_score
        CHECK (review_score BETWEEN 1 AND 5)
);


-- =========================================================
-- GEOLOCATION
-- =========================================================
-- Zip prefixes repeat (many coordinates per prefix), so the
-- table gets a surrogate key and no FKs point to it.

CREATE TABLE IF NOT EXISTS public.geolocation (
    geolocation_id BIGSERIAL PRIMARY KEY,
    geolocation_zip_code_prefix VARCHAR(5) NOT NULL,
    geolocation_lat DOUBLE PRECISION,
    geolocation_lng DOUBLE PRECISION,
    geolocation_city VARCHAR(100),
    geolocation_state CHAR(2)
);


-- =========================================================
-- INDEXES
-- =========================================================

CREATE INDEX IF NOT EXISTS idx_customers_unique_id
ON public.customers(customer_unique_id);

CREATE INDEX IF NOT EXISTS idx_customers_zip
ON public.customers(customer_zip_code_prefix);

CREATE INDEX IF NOT EXISTS idx_sellers_zip
ON public.sellers(seller_zip_code_prefix);

CREATE INDEX IF NOT EXISTS idx_products_category
ON public.products(product_category_name);

CREATE INDEX IF NOT EXISTS idx_orders_customer_id
ON public.orders(customer_id);

CREATE INDEX IF NOT EXISTS idx_orders_purchase_ts
ON public.orders(order_purchase_timestamp);

CREATE INDEX IF NOT EXISTS idx_orders_status
ON public.orders(order_status);

CREATE INDEX IF NOT EXISTS idx_order_items_product_id
ON public.order_items(product_id);

CREATE INDEX IF NOT EXISTS idx_order_items_seller_id
ON public.order_items(seller_id);

CREATE INDEX IF NOT EXISTS idx_order_reviews_order_id
ON public.order_reviews(order_id);

CREATE INDEX IF NOT EXISTS idx_geolocation_zip
ON public.geolocation(geolocation_zip_code_prefix);

"""

cursor.execute(create_tables_sql)

print("Tables created successfully")


# ============================================================
# OPTIONAL: CLEAR EXISTING DATA
# ============================================================

# Every execution completely reloads the CSV data.
# RESTART IDENTITY resets the geolocation_id counter.

cursor.execute("""
    TRUNCATE TABLE
        public.order_reviews,
        public.order_payments,
        public.order_items,
        public.orders,
        public.products,
        public.sellers,
        public.customers,
        public.product_category_name_translation,
        public.geolocation
    RESTART IDENTITY CASCADE;
""")


# ============================================================
# LOAD CSV USING POSTGRES COPY
# ============================================================

def load_csv(table_name, csv_file, columns):

    file_path = os.path.join(CSV_DIR, csv_file)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"CSV file not found: {file_path}"
        )

    copy_sql = sql.SQL("""
        COPY {} ({})
        FROM STDIN
        WITH (
            FORMAT CSV,
            HEADER TRUE,
            DELIMITER ',',
            NULL ''
        )
    """).format(
        sql.Identifier("public", table_name),
        sql.SQL(", ").join(
            sql.Identifier(column)
            for column in columns
        )
    )

    # utf-8-sig strips the byte-order mark at the start of
    # product_category_name_translation.csv (harmless for
    # the other files).
    with open(
        file_path,
        "r",
        encoding="utf-8-sig"
    ) as file:

        cursor.copy_expert(
            copy_sql,
            file
        )

    print(f"Loaded {csv_file}")


# ============================================================
# LOAD PRODUCT CATEGORY NAME TRANSLATION
# ============================================================

load_csv(
    "product_category_name_translation",
    "product_category_name_translation.csv",
    [
        "product_category_name",
        "product_category_name_english",
    ],
)


# ============================================================
# LOAD CUSTOMERS
# ============================================================

load_csv(
    "customers",
    "olist_customers_dataset.csv",
    [
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    ],
)


# ============================================================
# LOAD SELLERS
# ============================================================

load_csv(
    "sellers",
    "olist_sellers_dataset.csv",
    [
        "seller_id",
        "seller_zip_code_prefix",
        "seller_city",
        "seller_state",
    ],
)


# ============================================================
# LOAD PRODUCTS
# ============================================================

load_csv(
    "products",
    "olist_products_dataset.csv",
    [
        "product_id",
        "product_category_name",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ],
)


# ============================================================
# LOAD ORDERS
# ============================================================

load_csv(
    "orders",
    "olist_orders_dataset.csv",
    [
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
)


# ============================================================
# LOAD ORDER ITEMS
# ============================================================

load_csv(
    "order_items",
    "olist_order_items_dataset.csv",
    [
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value",
    ],
)


# ============================================================
# LOAD ORDER PAYMENTS
# ============================================================

load_csv(
    "order_payments",
    "olist_order_payments_dataset.csv",
    [
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value",
    ],
)


# ============================================================
# LOAD ORDER REVIEWS
# ============================================================

load_csv(
    "order_reviews",
    "olist_order_reviews_dataset.csv",
    [
        "review_id",
        "order_id",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp",
    ],
)


# ============================================================
# LOAD GEOLOCATION (~1M rows)
# ============================================================

load_csv(
    "geolocation",
    "olist_geolocation_dataset.csv",
    [
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
        "geolocation_city",
        "geolocation_state",
    ],
)


# ============================================================
# VERIFY RECORD COUNTS
# ============================================================

tables = [
    "product_category_name_translation",
    "customers",
    "sellers",
    "products",
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
    "geolocation",
]

print("\nRecord counts:")
print("-" * 50)

for table in tables:

    cursor.execute(
        sql.SQL(
            "SELECT COUNT(*) FROM {}.{}"
        ).format(
            sql.Identifier("public"),
            sql.Identifier(table)
        )
    )

    count = cursor.fetchone()[0]

    print(f"{table:<35} {count:>12,}")


# ============================================================
# COMMIT
# ============================================================

conn.commit()

print("\nData loaded successfully!")
print("Transaction committed.")


# ============================================================
# CLOSE CONNECTION
# ============================================================

cursor.close()
conn.close()

print("PostgreSQL connection closed.")