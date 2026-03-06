#!/bin/bash

# Set database and table names
DB_NAME="llm_logs"
TABLE_NAME="llm_logs"

# Function to check if a table exists
table_exists() {
    psql -h localhost -U postgres -d "$DB_NAME" -tAc "SELECT 1 FROM pg_tables WHERE tablename='$TABLE_NAME';" | grep -q 1
}

# Function to add a column if it doesn't exist
add_column_if_not_exists() {
    local column_name="$1"
    local column_type="$2"
    if ! psql -h localhost -U postgres -d "$DB_NAME" -tAc "SELECT 1 FROM information_schema.columns WHERE table_name='$TABLE_NAME' AND column_name='$column_name';" | grep -q 1; then
        echo "Adding column $column_name to $TABLE_NAME"
        psql -h localhost -U postgres -d "$DB_NAME" -c "ALTER TABLE $TABLE_NAME ADD COLUMN $column_name $column_type;"
    else
        echo "Column $column_name already exists in $TABLE_NAME"
    fi
}

# Check if the table exists
if table_exists; then
    echo "Table $TABLE_NAME exists. Checking columns..."
    # Add columns if they don't exist
    add_column_if_not_exists "input_token_count" "INTEGER"
    add_column_if_not_exists "output_token_count" "INTEGER"
    add_column_if_not_exists "latency_ms" "INTEGER"
    add_column_if_not_exists "is_json" "BOOLEAN"
    add_column_if_not_exists "prompt_type" "TEXT"
    add_column_if_not_exists "model_version" "TEXT"
    add_column_if_not_exists "error_message" "TEXT"
    add_column_if_not_exists "is_successful" "BOOLEAN"
    add_column_if_not_exists "user_id" "TEXT"
    add_column_if_not_exists "session_id" "TEXT"
    add_column_if_not_exists "user_feedback" "TEXT"
    add_column_if_not_exists "feedback_score" "INTEGER"
    add_column_if_not_exists "input_cost" "FLOAT"
    add_column_if_not_exists "output_cost" "FLOAT"
    add_column_if_not_exists "response_time" "TIMESTAMP"
else
    echo "Table $TABLE_NAME does not exist. Creating with all columns..."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        CREATE TABLE IF NOT EXISTS llm_logs (
            id SERIAL PRIMARY KEY,
            input_data TEXT NOT NULL,
            output_data TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            input_token_count INTEGER,
            output_token_count INTEGER,
            latency_ms INTEGER,
            is_json BOOLEAN,
            prompt_type TEXT,
            model_version TEXT,
            error_message TEXT,
            is_successful BOOLEAN,
            user_id TEXT,
            session_id TEXT,
            user_feedback TEXT,
            feedback_score INTEGER,
            input_cost FLOAT,
            output_cost FLOAT,
            response_time TIMESTAMP
        );
EOSQL
fi