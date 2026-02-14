# 1. Syntax Categories and Data Types

### 1. SQL Syntax Categories
| Category | Full Name | Purpose | Common Syntax |
|---|---|---|---|
| DDL | Data Definition Language | Define/modify database and table structures | CREATE, ALTER, DROP, TRUNCATE |
| DML | Data Manipulation Language | Insert, update, delete data in tables | INSERT, UPDATE, DELETE |
| DQL | Data Query Language | Query data | SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT |
| DCL | Data Control Language | User and privilege control | GRANT, REVOKE |
| TCL | Transaction Control Language | Transaction control | BEGIN / START TRANSACTION, COMMIT, ROLLBACK |

### 2. SQL Data Types
| Category | Type | Bytes | Range | Typical Usage |
|---|---|---|---|---|
| Numeric | TINYINT | 1B | -128 ~ 127 | status flags |
| Numeric | TINYINT UNSIGNED | 1B | 0 ~ 255 | counters, small IDs |
| Numeric | SMALLINT | 2B | -32,768 ~ 32,767 | small-range values |
| Numeric | SMALLINT UNSIGNED | 2B | 0 ~ 65,535 | age, quantity |
| Numeric | INT / INTEGER | 4B | $-2^{31} \sim 2^{31}-1$ | common integers |
| Numeric | INT UNSIGNED | 4B | $0 \sim 2^{32}-1$ | primary key IDs |
| Numeric | BIGINT | 8B | $-2^{63} \sim 2^{63}-1$ | large integers |
| Numeric | BIGINT UNSIGNED | 8B | $0 \sim 2^{64}-1$ | user IDs, order IDs |
| Numeric | FLOAT | 4B | about ±3.4E38 | probability, score |
| Numeric | DOUBLE | 8B | about ±1.7E308 | calculation results |
| Numeric | DECIMAL(p,s) | by precision | exact numeric | money |
| String | CHAR(n) | n chars | fixed length | fixed codes |
| String | VARCHAR(n) | ≤ n chars | variable length | names, accounts |
| String | TEXT | ≤ 65,535 chars | long text | remarks |
| Time | DATE | 3B | 1000-01-01 ~ 9999-12-31 | date |
| Time | TIME | 3B | -838:59:59 ~ 838:59:59 | time |
| Time | DATETIME | 8B | 1000-01-01 ~ 9999-12-31 | business timestamp |
| Time | TIMESTAMP | 4B | 1970-01-01 ~ 2038-01-19 | created/updated time |
| Boolean | BOOLEAN / BOOL | 1B | 0 / 1 | active/inactive |
| Enum | ENUM | 1–2B | predefined values | status |

# 2. DDL
### 1. Database Operations
- List all databases (DQL)
    ```sql
    SHOW DATABASES;
    ```
- Show current database (DQL)
    ```sql
    SELECT DATABASE();
    ```
- Create database
    ```sql
    CREATE DATABASE db_name;
    -- minimal form
    ```
    ```sql
    CREATE DATABASE (IF NOT EXISTS) db_name (DEFAULT CHARSET charset) (COLLATE collation);
    -- IF NOT EXISTS avoids errors; charset/collation are optional.
    ```

### 2. Table Operations
- Show all tables in current database (DQL)
    ```sql
    SHOW TABLES;
    ```
- Show table structure (DQL)
    ```sql
    DESC table_name;
    ```
- Show CREATE TABLE statement (DQL)
    ```sql
    SHOW CREATE TABLE table_name;
    ```
- Create table
    ```sql
    CREATE TABLE table_name(
        col1 type,
        col2 type
    );
    ```
    ```sql
    CREATE TABLE table_name(
        col1 type COMMENT 'column comment',
        col2 type COMMENT 'column comment'
    ) COMMENT = 'table comment';
    ```
- Drop table
    ```sql
    DROP TABLE table_name;
    -- DROP TABLE IF EXISTS table_name to avoid errors
    -- DROP TABLE table1, table2, table3
    ```
- Truncate table data
    ```sql
    TRUNCATE TABLE table_name;
    -- IF EXISTS is not supported; table attributes remain unchanged
    ```
- Rename table
    ```sql
    ALTER TABLE old_table_name RENAME TO new_table_name;
    -- attributes remain unchanged
    ```

### 3. Table Column Operations
- Add column
    ```sql
    ALTER TABLE table_name ADD column_name data_type;
    -- you can append COMMENT and constraints
    ```
- Drop column
    ```sql
    ALTER TABLE table_name DROP column_name;
    -- also written as DROP COLUMN column_name
    ```
- Modify data type
    ```sql
    ALTER TABLE table_name
    MODIFY column_name new_data_type;
    -- may overwrite attributes; append NOT NULL / DEFAULT / COMMENT if needed
    ```
- Modify column name and data type
    ```sql
    ALTER TABLE table_name
    CHANGE old_column_name new_column_name new_data_type;
    -- CHANGE may also overwrite attributes
    ```

# 3. DML
### 1. Insert Data
- Insert into specific columns
    ```sql
    INSERT INTO table_name (col1, col2)
    VALUES (v1, v2);
    -- string/date values need quotes
    ```
- Insert into all columns
    ```sql
    INSERT INTO table_name
    VALUES (v1, v2, v3);
    ```
- Batch insert
    ```sql
    INSERT INTO table_name (col1, col2)
    VALUES (v1, v2),
           (v1, v2),
           (v1, v2);
    ```
    ```sql
    INSERT INTO table_name
    VALUES (v1, v2, v3),
           (v1, v2, v3),
           (v1, v2, v3);
    ```

### 2. Update Data
- Update all rows
    ```sql
    UPDATE table_name SET col1 = v1, col2 = v2;
    ```
- Update specific rows
    ```sql
    UPDATE table_name SET col1 = v1, col2 = v2 WHERE condition;
    ```
- Delete rows
    ```sql
    DELETE FROM table_name WHERE condition;
    -- if WHERE is omitted, all rows are deleted
    ```

# 4. DQL
### 1. DQL Syntax
    ```sql
    SELECT      columns
    FROM        table_name
    WHERE       conditions
    GROUP BY    grouping_columns
    HAVING      post_group_conditions
    ORDER BY    sort_columns
    LIMIT       pagination
    ```

### 2. Basic DQL Queries
- Query multiple columns
    ```sql
    SELECT col1, col2, col3 FROM table_name;
    ```
    ```sql
    SELECT * FROM table_name;
    ```
- Set column aliases
    ```sql
    SELECT col1 AS alias1, col2 AS alias2 FROM table_name;
    -- AS can be omitted
    ```
- Remove duplicates
    ```sql
    SELECT DISTINCT col FROM table_name;
    -- DISTINCT col1, col2 means unique combinations of two columns
    ```

### 3. Conditional Query
- Syntax
    ```sql
    SELECT columns FROM table_name WHERE condition;
    ```
- Conditions
    | Type | Condition | Meaning | Example |
    |---|---|---|---|
    | Comparison | = | equal | price = 100 |
    | Comparison | != / <> | not equal | price != 100 |
    | Comparison | > | greater than | price > 100 |
    | Comparison | < | less than | price < 100 |
    | Comparison | >= | greater than or equal | price >= 100 |
    | Comparison | <= | less than or equal | price <= 100 |
    | Range | BETWEEN a AND b | within range (inclusive) | price BETWEEN 100 AND 200 |
    | Range | NOT BETWEEN | out of range | price NOT BETWEEN 100 AND 200 |
    | Set | IN (...) | in set | brand IN ('BMW','Audi') |
    | Set | NOT IN (...) | not in set | brand NOT IN ('BMW','Audi') |
    | Null | IS NULL | is null | discount IS NULL |
    | Null | IS NOT NULL | not null | discount IS NOT NULL |
    | Fuzzy | LIKE '%x%' | contains | model LIKE '%X%' |
    | Fuzzy | LIKE 'x%' | starts with x | model LIKE 'X%' |
    | Fuzzy | LIKE '%x' | ends with x | model LIKE '%X' |
    | Fuzzy | NOT LIKE | not match | model NOT LIKE '%X%' |

- Logical operators

    | Operator | Meaning | Description | Example |
    |---|---|---|---|
    | AND / && | and | all conditions true | price > 100 AND brand = 'BMW' |
    | OR / \|\| | or | any condition true | brand = 'BMW' OR brand = 'Audi' |
    | NOT / ! | not | negate condition | NOT price > 100 |
    | XOR | exclusive or | only one condition true | brand = 'BMW' XOR brand = 'Audi' |

- Aggregate functions
    | Function | Meaning | Scope | Example | Notes |
    |---|---|---|---|---|
    | COUNT(*) | count rows | rows | COUNT(*) | includes NULL rows |
    | COUNT(col) | count non-null values | column | COUNT(price) | ignores NULL |
    | SUM(col) | sum values | numeric column | SUM(price) | ignores NULL |
    | AVG(col) | average | numeric column | AVG(price) | ignores NULL |
    | MAX(col) | maximum | column | MAX(price) | works on strings too |
    | MIN(col) | minimum | column | MIN(price) | works on strings too |

### 4. Group Query
    ```sql
    SELECT columns FROM table_name (WHERE condition) GROUP BY group_column (HAVING post_group_filter)
    ```
    - `WHERE` filters before grouping; `HAVING` filters after grouping.
    - `WHERE` cannot use aggregate functions, `HAVING` can.
    - Execution order: `WHERE` > aggregate functions > `HAVING`.
    - After grouping, typically query only group columns and aggregate functions.

### 5. Sorting and Pagination Query
- Sorting query
    ```sql
    SELECT columns FROM table_name
    ORDER BY col1 ASC, col2 DESC;
    -- ASC ascending (default, can be omitted); DESC descending.
    ```
- Pagination query
    ```sql
    SELECT columns FROM table_name LIMIT offset, row_count;
    ```
    - `offset = rows_per_page * (page_number - 1)`
    - first page can be written as `LIMIT 10`

### 6. Execution Order of DQL
    ```sql
    FROM - WHERE - GROUP BY - HAVING - SELECT - ORDER BY - LIMIT
    ```

# 5. DCL
### 1. User Management
- Query users
    ```sql
    USE mysql;
    SELECT * FROM user;
    -- user info is stored in mysql.user
    ```
- Create user
    ```sql
    CREATE USER 'username'@'hostname' IDENTIFIED BY 'password';
    -- localhost means local machine
    -- '%' means any host
    ```
- Delete user
    ```sql
    DROP USER 'username@hostname';
    ```
- Modify user password
    ```sql
    ALTER USER 'username@hostname'
    IDENTIFIED WITH caching_sha2_password BY 'new_password';
    -- before MySQL 8.0: mysql_native_password is commonly used
    ```

### 2. User Privilege Control
- Query privileges
    ```sql
    SHOW GRANTS FOR 'username'@'hostname';
    ```
- Grant privileges
    ```sql
    GRANT privileges ON database.table TO 'username'@'hostname';
    -- all privileges: ALL PRIVILEGES (PRIVILEGES can be omitted)
    -- all databases/tables: *.*
    ```
- Revoke privileges
    ```sql
    REVOKE privileges ON database.table TO 'username'@'hostname';
    ```
- Privilege list
    | Privilege | Description | Typical Use |
    |------|---------|-------------|
    | ALL / ALL PRIVILEGES | all privileges | admin / root |
    | USAGE | no privilege (account exists only) | default state |
    | SELECT | query data | read-only user |
    | INSERT | insert data | data entry |
    | UPDATE | update data | modify data |
    | DELETE | delete data | delete records |
    | CREATE | create databases/tables | schema creation |
    | DROP | drop databases/tables | delete schema |
    | ALTER | alter table structure | change columns |
    | INDEX | create/drop indexes | performance tuning |
    | CREATE VIEW | create views | reports |
    | SHOW VIEW | view definition | ops/inspection |
    | TRIGGER | create triggers | advanced logic |
    | EXECUTE | execute stored procedures | procedure calls |
    | EVENT | create scheduler events | scheduled jobs |
    | REFERENCES | foreign key constraints | table relations |
    | GRANT OPTION | grant owned privileges to others | delegated admin |

# 6. Functions

### 1. String Functions

    | Function | Purpose | Example | Result |
    |------|------|------|------|
    | LENGTH(str) | byte length | LENGTH('abc') | 3 |
    | CHAR_LENGTH(str) | character count | CHAR_LENGTH('你好') | 2 |
    | CONCAT(str1, str2, ...) | concatenate | CONCAT('a','b','c') | abc |
    | CONCAT_WS(sep, str1, ...) | concatenate with separator | CONCAT_WS('-', '2024','01','01') | 2024-01-01 |
    | UPPER(str) | uppercase | UPPER('abc') | ABC |
    | LOWER(str) | lowercase | LOWER('ABC') | abc |
    | LEFT(str, n) | left n chars | LEFT('abcdef',3) | abc |
    | RIGHT(str, n) | right n chars | RIGHT('abcdef',3) | def |
    | SUBSTRING(str, start, len) | substring | SUBSTRING('abcdef',2,3) | bcd |
    | SUBSTR(str, start, len) | alias of SUBSTRING | SUBSTR('abcdef',2,3) | bcd |
    | TRIM(str) | trim both sides | TRIM('  hi  ') | hi |
    | LTRIM(str) | trim left | LTRIM('  hi') | hi |
    | RTRIM(str) | trim right | RTRIM('hi  ') | hi |
    | REPLACE(str, from, to) | replace text | REPLACE('a-b-c','-','+') | a+b+c |
    | INSTR(str, sub) | find position | INSTR('hello','e') | 2 |
    | LOCATE(sub, str) | find position | LOCATE('e','hello') | 2 |
    | LPAD(str, len, pad) | left pad | LPAD('5',3,'0') | 005 |
    | RPAD(str, len, pad) | right pad | RPAD('5',3,'0') | 500 |
    | REVERSE(str) | reverse string | REVERSE('abc') | cba |
    | STRCMP(str1, str2) | compare strings | STRCMP('a','b') | -1 |
    | FIND_IN_SET(str, list) | index in set | FIND_IN_SET('b','a,b,c') | 2 |

### 2. Extracting Functions

    | Function | Syntax | Meaning | Key Rule | Example | Result |
    |----|----|----|----|----|----|
    | LEFT | LEFT(str, n) | extract n from left | n ≤ 0 returns empty string | LEFT('abcdef', 3) | abc |
    | RIGHT | RIGHT(str, n) | extract n from right | n ≤ 0 returns empty string | RIGHT('abcdef', 2) | ef |
    | SUBSTRING | SUBSTRING(str, pos, len) | extract len from pos | pos starts from 1 | SUBSTRING('abcdef', 2, 3) | bcd |
    | SUBSTRING | SUBSTRING(str, pos) | from pos to end | pos can be negative | SUBSTRING('abcdef', 3) | cdef |
    | SUBSTRING | SUBSTRING(str, -n) | n from right | same as RIGHT | SUBSTRING('abcdef', -2) | ef |
    | MID | MID(str, pos, len) | alias of SUBSTRING | equivalent | MID('abcdef', 2, 2) | bc |
    | SUBSTR | SUBSTR(str, pos, len) | alias of SUBSTRING | equivalent | SUBSTR('abcdef', 1, 4) | abcd |
    | CHAR_LENGTH | CHAR_LENGTH(str) | char count | Chinese char counts as 1 | CHAR_LENGTH('你好a') | 3 |
    | LENGTH | LENGTH(str) | byte count | UTF-8 Chinese char counts as 3 | LENGTH('你好a') | 7 |

### 3. Date Functions
    | Function | Syntax | Meaning | Key Rule | Example | Result |
    |----|----|----|----|----|----|
    | CURDATE | CURDATE() | current date | no time part | CURDATE() | 2026-01-21 |
    | CURRENT_DATE | CURRENT_DATE | current date | same as CURDATE | CURRENT_DATE | 2026-01-21 |
    | NOW | NOW() | current datetime | includes time | NOW() | 2026-01-21 14:xx:xx |
    | SYSDATE | SYSDATE() | system current time | not transaction-consistent | SYSDATE() | 2026-01-21 14:xx:xx |
    | CURRENT_TIMESTAMP | CURRENT_TIMESTAMP | current timestamp | same as NOW | CURRENT_TIMESTAMP | 2026-01-21 14:xx:xx |
    | DATE | DATE(expr) | get date part | remove time | DATE('2026-01-21 10:30:00') | 2026-01-21 |
    | TIME | TIME(expr) | get time part | remove date | TIME('2026-01-21 10:30:00') | 10:30:00 |
    | YEAR | YEAR(date) | get year | integer | YEAR('2026-01-21') | 2026 |
    | MONTH | MONTH(date) | get month | 1–12 | MONTH('2026-01-21') | 1 |
    | DAY | DAY(date) | get day | 1–31 | DAY('2026-01-21') | 21 |
    | HOUR | HOUR(time) | get hour | 0–23 | HOUR('14:30:00') | 14 |
    | MINUTE | MINUTE(time) | get minute | 0–59 | MINUTE('14:30:45') | 30 |
    | SECOND | SECOND(time) | get second | 0–59 | SECOND('14:30:45') | 45 |
    | DAYOFWEEK | DAYOFWEEK(date) | weekday | Sunday=1 | DAYOFWEEK('2026-01-21') | 4 |
    | WEEKDAY | WEEKDAY(date) | weekday | Monday=0 | WEEKDAY('2026-01-21') | 2 |
    | DAYOFMONTH | DAYOFMONTH(date) | day in month | same as DAY | DAYOFMONTH('2026-01-21') | 21 |
    | DAYOFYEAR | DAYOFYEAR(date) | day in year | 1–366 | DAYOFYEAR('2026-01-21') | 21 |
    | LAST_DAY | LAST_DAY(date) | last day of month | returns date | LAST_DAY('2026-02-01') | 2026-02-28 |
    | DATE_ADD | DATE_ADD(date, INTERVAL n unit) | date add | supports multiple units | DATE_ADD('2026-01-21', INTERVAL 3 DAY) | 2026-01-24 |
    | DATE_SUB | DATE_SUB(date, INTERVAL n unit) | date subtract | same | DATE_SUB('2026-01-21', INTERVAL 1 MONTH) | 2025-12-21 |
    | DATEDIFF | DATEDIFF(d1, d2) | date diff | d1 - d2 in days | DATEDIFF('2026-01-21','2026-01-01') | 20 |
    | TIMESTAMPDIFF | TIMESTAMPDIFF(unit, t1, t2) | datetime diff | unit-based precision | TIMESTAMPDIFF(DAY,'2026-01-01','2026-01-21') | 20 |
    | STR_TO_DATE | STR_TO_DATE(str, fmt) | string to date | format required | STR_TO_DATE('2026-01-21','%Y-%m-%d') | 2026-01-21 |
    | DATE_FORMAT | DATE_FORMAT(date, fmt) | date to string | common formatting | DATE_FORMAT('2026-01-21','%Y/%m/%d') | 2026/01/21 |

### 4. Flow Control Functions
    | Function | Purpose | Syntax | Example | Result |
    |---|---|---|---|---|
    | IF(expr, v1, v2) | conditional choice | IF(cond, true_val, false_val) | IF(60>=60,'pass','fail') | pass |
    | IFNULL(v1, v2) | return v2 when v1 is NULL | IFNULL(v1, v2) | IFNULL(NULL,0) | 0 |
    | NULLIF(v1, v2) | return NULL when equal | NULLIF(v1,v2) | NULLIF(5,5) | NULL |
    | CASE WHEN | multi-branch conditions | CASE WHEN cond THEN val END | CASE WHEN score>=60 THEN 'pass' END | pass |
    | CASE expr WHEN | equality branching | CASE expr WHEN val THEN val END | CASE sex WHEN 'M' THEN 1 END | 1 |

# 7. Constraints
### 1. Constraint Categories
    | Constraint | Purpose | Position | Example | Notes |
    |---|---|---|---|---|
    | PRIMARY KEY | unique + not null key | column/table | id INT PRIMARY KEY | only one per table |
    | UNIQUE | uniqueness constraint | column/table | email VARCHAR(50) UNIQUE | can have multiple |
    | NOT NULL | non-null constraint | column | name VARCHAR(20) NOT NULL | NULL forbidden |
    | DEFAULT | default value | column | status INT DEFAULT 1 | can be omitted on insert |
    | CHECK | conditional constraint | column/table | age INT CHECK (age >= 0) | MySQL 8+ effective |
    | FOREIGN KEY | foreign key | table | FOREIGN KEY (cid) REFERENCES class(id) | table relation |
    | AUTO_INCREMENT | auto increment | column | id INT AUTO_INCREMENT | often with PK |

### 2. Primary Key and Foreign Key
- Add primary key after table creation
    ```sql
    ALTER TABLE table_name
    ADD PRIMARY KEY (pk_column);
    -- PK implies UNIQUE + NOT NULL
    ```
- Drop primary key
    ```sql
    ALTER TABLE table_name
    DROP PRIMARY KEY;
    ```
- View primary key info
    ```sql
    SHOW KEYS FROM table_name WHERE Key_name = 'PRIMARY';
    ```
- Define foreign key during table creation
    ```sql
    CREATE TABLE table_name (
    ...
    ...
    ...

    CONSTRAINT fk_name
    FOREIGN KEY (fk_column) REFERENCES parent_table(parent_column)
    );
    -- CONSTRAINT name is optional, but named FK is easier to drop later
    ```
- Add foreign key after table creation
    ```sql
    ALTER TABLE table_name
    ADD CONSTRAINT fk_name
    FOREIGN KEY (fk_column) REFERENCES parent_table(parent_column)
    ```

### 3. Foreign Key Constraints
- RESTRICT / NO ACTION (default)
    ```sql
    ALTER TABLE table_name ADD CONSTRAINT fk_name
    FOREIGN KEY (fk_column) REFERENCES parent_table(parent_column)
    ON UPDATE RESTRICT ON DELETE RESTRICT
    -- NO ACTION behaves like RESTRICT.
    -- If child rows reference parent row, update/delete on parent is blocked.
    ```
- CASCADE
    ```sql
    ON UPDATE CASCADE
    ON DELETE CASCADE
    -- parent updates/deletes propagate to child rows
    ```
- SET NULL
    ```sql
    ON UPDATE SET NULL
    ON DELETE SET NULL
    -- child FK set to NULL when parent changes/deletes (FK must allow NULL)
    ```
- SET DEFAULT: not supported in MySQL.

### 4. Process for Modifying FK Constraints
- 1. Check foreign key name
    ```sql
    SHOW CREATE TABLE orders;
    ```
- 2. Drop original foreign key
    ```sql
    ALTER TABLE table_name
    DROP FOREIGN KEY fk_name;
    ```
- 3. Re-add foreign key
    ```sql
    ALTER TABLE table_name
    ADD CONSTRAINT fk_name
    FOREIGN KEY (fk_column) REFERENCES parent_table(parent_column)
    ON DELETE SET NULL ON UPDATE SET NULL;
    ```

# 8. Multi-table Query
### 1. Inner Join
- Implicit inner join
    ```sql
    SELECT column_list FROM table1, table2
    WHERE join_condition;
    -- columns should be prefixed by table name, e.g., Students.student_id
    ```
- Explicit inner join
    ```sql
    SELECT column_list FROM table1
    INNER JOIN table2 ON join_condition;
    -- INNER can be omitted
    ```
* Use table aliases to simplify joins; after aliasing, use alias instead of original table name.
* Column aliases do not change how source tables are referenced.

### 2. Outer Join
- Left outer join
    ```sql
    SELECT column_list FROM table1
    LEFT OUTER JOIN table2 ON condition;
    -- OUTER can be omitted
    -- returns all rows from table1 + intersection with table2; unmatched values are NULL
    ```
- Right outer join
    ```sql
    SELECT column_list FROM table1
    RIGHT JOIN table2 ON condition;
    -- returns all rows from table2 + intersection
    ```

### 3. Self Join
    ```sql
    SELECT column_list FROM table1 alias1
    JOIN table1 alias2 ON condition;
    -- alias is required in self join
    -- can be INNER / LEFT / RIGHT join
    ```

### 4. Union Query
- Merge results of multiple queries
    ```sql
    SELECT columns FROM ...
    UNION ALL
    SELECT columns FROM ...
    -- UNION ALL keeps duplicates; UNION removes duplicates
    -- both SELECTs must have same column count and compatible types
    -- only one ORDER BY at the end
    ```

### 5. Subquery
- Nested query
- Scalar subquery
    > returns a single value (number/string/time), commonly used with =, <>, >, >=, <, <=
- Column subquery
    > returns one column, commonly used with IN, NOT IN, ANY, SOME, ALL
    - **IN**: in the set
    - **NOT IN**: not in the set
    - **ANY / SOME**: satisfies at least one
    - **ALL**: satisfies all
    - **EXIST**: passes if exists
- Row subquery
    > returns one row, commonly used with =, <>, IN, NOT IN

# 9. Transactions
    A transaction is a group of SQL operations treated as one indivisible unit. Operations are either committed together or rolled back together.

- Check auto-commit mode
    ```sql
    SELECT @@autocommit;
    ```
- Set commit mode
    ```sql
    SET @@autocommit = 0;
    -- 1 = auto commit, 0 = manual commit
    ```
- Commit transaction
    ```sql
    COMMIT;
    ```
- Rollback transaction
    ```sql
    ROLLBACK;
    ```
- Start transaction
    ```sql
    START / BEGIN TRANSACTION;
    ```

# To be updated...
