# 一、文法分類とデータ型

### 1. SQL文法の分類
| 分類 | 正式名称 | 役割 | よく使う構文 |
|---|---|---|---|
| DDL | Data Definition Language | DB/テーブル構造の定義・変更 | CREATE、ALTER、DROP、TRUNCATE |
| DML | Data Manipulation Language | テーブルデータの追加・更新・削除 | INSERT、UPDATE、DELETE |
| DQL | Data Query Language | データ検索 | SELECT、FROM、WHERE、GROUP BY、HAVING、ORDER BY、LIMIT |
| DCL | Data Control Language | 権限とユーザー管理 | GRANT、REVOKE |
| TCL | Transaction Control Language | トランザクション制御 | BEGIN / START TRANSACTION、COMMIT、ROLLBACK |

### 2. SQLデータ型
| 分類 | 型 | バイト | 範囲 | 主な用途 |
|---|---|---|---|---|
| 数値 | TINYINT | 1B | -128 ~ 127 | 状態フラグ |
| 数値 | TINYINT UNSIGNED | 1B | 0 ~ 255 | カウント、小ID |
| 数値 | SMALLINT | 2B | -32,768 ~ 32,767 | 小範囲数値 |
| 数値 | SMALLINT UNSIGNED | 2B | 0 ~ 65,535 | 年齢、数量 |
| 数値 | INT / INTEGER | 4B | $-2^{31} \sim 2^{31}-1$ | 一般整数 |
| 数値 | INT UNSIGNED | 4B | $0 \sim 2^{32}-1$ | 主キーID |
| 数値 | BIGINT | 8B | $-2^{63} \sim 2^{63}-1$ | 大きい整数 |
| 数値 | BIGINT UNSIGNED | 8B | $0 \sim 2^{64}-1$ | ユーザーID、注文ID |
| 数値 | FLOAT | 4B | 約 ±3.4E38 | 確率、スコア |
| 数値 | DOUBLE | 8B | 約 ±1.7E308 | 計算結果 |
| 数値 | DECIMAL(p,s) | 精度依存 | 正確表現 | 金額 |
| 文字列 | CHAR(n) | n文字 | 固定長 | 固定コード |
| 文字列 | VARCHAR(n) | ≤ n文字 | 可変長 | 名前、アカウント |
| 文字列 | TEXT | ≤ 65,535文字 | 長文 | 備考 |
| 時間 | DATE | 3B | 1000-01-01 ~ 9999-12-31 | 日付 |
| 時間 | TIME | 3B | -838:59:59 ~ 838:59:59 | 時刻 |
| 時間 | DATETIME | 8B | 1000-01-01 ~ 9999-12-31 | 業務時刻 |
| 時間 | TIMESTAMP | 4B | 1970-01-01 ~ 2038-01-19 | 作成/更新時刻 |
| 真偽 | BOOLEAN / BOOL | 1B | 0 / 1 | 有効/無効 |
| 列挙 | ENUM | 1–2B | 事前定義値 | 状態 |

# 二、DDL
### 1. データベース操作
- すべてのDBを表示 (DQL)
    ```sql
    SHOW DATABASES;
    ```
- 現在のDBを表示 (DQL)
    ```sql
    SELECT DATABASE();
    ```
- DB作成
    ```sql
    CREATE DATABASE db_name;
    -- 最小構文
    ```
    ```sql
    CREATE DATABASE (IF NOT EXISTS) db_name (DEFAULT CHARSET charset) (COLLATE collation);
    -- IF NOT EXISTSでエラー回避。文字コード/照合順序は任意。
    ```

### 2. テーブル操作
- 現在DB内の全テーブルを表示 (DQL)
    ```sql
    SHOW TABLES;
    ```
- テーブル構造を表示 (DQL)
    ```sql
    DESC table_name;
    ```
- CREATE文を表示 (DQL)
    ```sql
    SHOW CREATE TABLE table_name;
    ```
- テーブル作成
    ```sql
    CREATE TABLE table_name(
        col1 type,
        col2 type
    );
    ```
    ```sql
    CREATE TABLE table_name(
        col1 type COMMENT '列コメント',
        col2 type COMMENT '列コメント'
    ) COMMENT = 'テーブルコメント';
    ```
- テーブル削除
    ```sql
    DROP TABLE table_name;
    -- DROP TABLE IF EXISTS table_name でエラー回避
    -- DROP TABLE table1,table2,table3
    ```

- テーブルデータ全削除
    ```sql
    TRUNCATE TABLE table_name;
    -- IF EXISTSは不可、属性は維持
    ```
- テーブル名変更
    ```sql
    ALTER TABLE old_table_name RENAME TO new_table_name;
    -- 属性は変わらない
    ```

### 3. カラム操作
- カラム追加
    ```sql
    ALTER TABLE table_name ADD column_name data_type;
    -- COMMENT/制約を後ろに追加可能
    ```
- カラム削除
    ```sql
    ALTER TABLE table_name DROP column_name;
    -- DROP COLUMN column_name と書く場合もある
    ```
- データ型変更
    ```sql
    ALTER TABLE table_name
    MODIFY column_name new_data_type;
    -- 既存属性を上書きしうるため、必要なら NOT NULL / DEFAULT / COMMENT を追記
    ```
- カラム名＋型変更
    ```sql
    ALTER TABLE table_name
    CHANGE old_column_name new_column_name new_data_type;
    -- CHANGE も属性を失う可能性あり
    ```

# 三、DML
### 1. データ追加
- 指定カラムへ追加
    ```sql
    INSERT INTO table_name (col1, col2)
    VALUES (v1, v2);
    -- 文字列/日付は引用符が必要
    ```
- 全カラムへ追加
    ```sql
    INSERT INTO table_name
    VALUES (v1, v2, v3);
    ```
- 一括追加
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

### 2. データ更新
- 全件更新
    ```sql
    UPDATE table_name SET col1=v1, col2=v2;
    ```
- 条件更新
    ```sql
    UPDATE table_name SET col1=v1, col2=v2 WHERE condition;
    ```
- データ削除
    ```sql
    DELETE FROM table_name WHERE condition;
    -- WHEREなしは全削除
    ```

# 四、DQL
### 1. DQL構文
    ```sql
    SELECT      columns
    FROM        table_name
    WHERE       conditions
    GROUP BY    grouping_columns
    HAVING      post_group_conditions
    ORDER BY    sort_columns
    LIMIT       pagination
    ```

### 2. DQL基本検索
- 複数カラム検索
    ```sql
    SELECT col1, col2, col3 FROM table_name;
    ```
    ```sql
    SELECT * FROM table_name;
    ```
- カラム別名設定
    ```sql
    SELECT col1 AS alias1, col2 AS alias2 FROM table_name;
    -- AS は省略可
    ```
- 重複排除
    ```sql
    SELECT DISTINCT col FROM table_name;
    -- DISTINCT col1,col2 は2カラムの組み合わせを一意化
    ```

### 3. 条件検索
- 構文
    ```sql
    SELECT columns FROM table_name WHERE condition;
    ```
- 条件一覧
    | 分類 | 条件 | 意味 | 例 |
    |---|---|---|---|
    | 比較 | = | 等しい | price = 100 |
    | 比較 | != / <> | 等しくない | price != 100 |
    | 比較 | > | より大きい | price > 100 |
    | 比較 | < | より小さい | price < 100 |
    | 比較 | >= | 以上 | price >= 100 |
    | 比較 | <= | 以下 | price <= 100 |
    | 範囲 | BETWEEN a AND b | 範囲内（両端含む） | price BETWEEN 100 AND 200 |
    | 範囲 | NOT BETWEEN | 範囲外 | price NOT BETWEEN 100 AND 200 |
    | 集合 | IN (...) | 集合に含む | brand IN ('BMW','Audi') |
    | 集合 | NOT IN (...) | 集合に含まない | brand NOT IN ('BMW','Audi') |
    | NULL | IS NULL | NULLである | discount IS NULL |
    | NULL | IS NOT NULL | NULLではない | discount IS NOT NULL |
    | 曖昧 | LIKE '%x%' | 含む | model LIKE '%X%' |
    | 曖昧 | LIKE 'x%' | xで始まる | model LIKE 'X%' |
    | 曖昧 | LIKE '%x' | xで終わる | model LIKE '%X' |
    | 曖昧 | NOT LIKE | 一致しない | model NOT LIKE '%X%' |

- 論理演算子

    | 演算子 | 意味 | 説明 | 例 |
    |---|---|---|---|
    | AND / && | かつ | 全条件が真 | price > 100 AND brand = 'BMW' |
    | OR / \|\| | または | いずれか真 | brand = 'BMW' OR brand = 'Audi' |
    | NOT / ! | 否定 | 条件反転 | NOT price > 100 |
    | XOR | 排他的論理和 | どちらか片方のみ真 | brand = 'BMW' XOR brand = 'Audi' |

- 集約関数
    | 関数 | 意味 | 対象 | 例 | 補足 |
    |---|---|---|---|---|
    | COUNT(*) | 行数 | 行 | COUNT(*) | NULL行も含む |
    | COUNT(col) | 非NULL数 | 列 | COUNT(price) | NULLは除外 |
    | SUM(col) | 合計 | 数値列 | SUM(price) | NULLは除外 |
    | AVG(col) | 平均 | 数値列 | AVG(price) | NULLは除外 |
    | MAX(col) | 最大 | 列 | MAX(price) | 文字列にも適用可 |
    | MIN(col) | 最小 | 列 | MIN(price) | 文字列にも適用可 |

### 4. グループ検索
    ```sql
    SELECT columns FROM table_name (WHERE condition) GROUP BY group_column (HAVING post_group_filter)
    ```
    - `WHERE` はグループ化前フィルタ、`HAVING` はグループ化後フィルタ。
    - `WHERE` は集約関数不可、`HAVING` は使用可。
    - 実行順序は `WHERE > 集約関数 > HAVING`。
    - グループ化後は通常「グループ列 + 集約関数」を選択。

### 5. 並び替え・ページング検索
- 並び替え
    ```sql
    SELECT columns FROM table_name
    ORDER BY col1 ASC, col2 DESC;
    -- ASC は昇順（既定、省略可）、DESC は降順
    ```
- ページング
    ```sql
    SELECT columns FROM table_name LIMIT offset, row_count;
    ```
    - `offset = 1ページ件数 * (ページ番号 - 1)`
    - 1ページ目は `LIMIT 10` のように書ける

### 6. DQL実行順序
    ```sql
    FROM - WHERE - GROUP BY - HAVING - SELECT - ORDER BY - LIMIT
    ```

# 五、DCL
### 1. ユーザー管理
- ユーザー確認
    ```sql
    USE mysql;
    SELECT * FROM user;
    -- ユーザー情報は mysql.user にある
    ```
- ユーザー作成
    ```sql
    CREATE USER 'username'@'hostname' IDENTIFIED BY 'password';
    -- localhost はローカル接続
    -- '%' は任意ホスト
    ```
- ユーザー削除
    ```sql
    DROP USER 'username@hostname';
    ```
- パスワード変更
    ```sql
    ALTER USER 'username@hostname'
    IDENTIFIED WITH caching_sha2_password BY 'new_password';
    -- MySQL 8.0以前は mysql_native_password が一般的
    ```

### 2. ユーザー権限制御
- 権限確認
    ```sql
    SHOW GRANTS FOR 'username'@'hostname';
    ```
- 権限付与
    ```sql
    GRANT privileges ON database.table TO 'username'@'hostname';
    -- 全権限は ALL PRIVILEGES（PRIVILEGES省略可）
    -- 全DB/全テーブルは *.*
    ```
- 権限剥奪
    ```sql
    REVOKE privileges ON database.table TO 'username'@'hostname';
    ```
- 権限一覧
    | 権限名 | 役割 | 主な場面 |
    |------|---------|-------------|
    | ALL / ALL PRIVILEGES | 全権限 | 管理者 / root |
    | USAGE | 権限なし（アカウント存在のみ） | 初期状態 |
    | SELECT | データ参照 | 読み取り専用ユーザー |
    | INSERT | データ追加 | 入力業務 |
    | UPDATE | データ更新 | 修正業務 |
    | DELETE | データ削除 | レコード削除 |
    | CREATE | DB/テーブル作成 | スキーマ作成 |
    | DROP | DB/テーブル削除 | 削除作業 |
    | ALTER | テーブル構造変更 | 列変更 |
    | INDEX | インデックス作成/削除 | 性能調整 |
    | CREATE VIEW | ビュー作成 | レポート |
    | SHOW VIEW | ビュー定義参照 | 運用確認 |
    | TRIGGER | トリガー作成 | 高度ロジック |
    | EXECUTE | ストアド実行 | 手続き実行 |
    | EVENT | イベント作成 | 定期処理 |
    | REFERENCES | 外部キー制約 | テーブル関連付け |
    | GRANT OPTION | 自分の権限を再付与可 | 権限委譲 |

# 六、関数

### 1. 文字列関数

    | 関数名 | 役割 | 例 | 結果 |
    |------|------|------|------|
    | LENGTH(str) | 文字列バイト長 | LENGTH('abc') | 3 |
    | CHAR_LENGTH(str) | 文字数 | CHAR_LENGTH('你好') | 2 |
    | CONCAT(str1, str2, ...) | 連結 | CONCAT('a','b','c') | abc |
    | CONCAT_WS(sep, str1, ...) | 区切り付き連結 | CONCAT_WS('-', '2024','01','01') | 2024-01-01 |
    | UPPER(str) | 大文字化 | UPPER('abc') | ABC |
    | LOWER(str) | 小文字化 | LOWER('ABC') | abc |
    | LEFT(str, n) | 左から n 文字 | LEFT('abcdef',3) | abc |
    | RIGHT(str, n) | 右から n 文字 | RIGHT('abcdef',3) | def |
    | SUBSTRING(str, start, len) | 部分文字列 | SUBSTRING('abcdef',2,3) | bcd |
    | SUBSTR(str, start, len) | SUBSTRING別名 | SUBSTR('abcdef',2,3) | bcd |
    | TRIM(str) | 両端空白除去 | TRIM('  hi  ') | hi |
    | LTRIM(str) | 左空白除去 | LTRIM('  hi') | hi |
    | RTRIM(str) | 右空白除去 | RTRIM('hi  ') | hi |
    | REPLACE(str, from, to) | 文字列置換 | REPLACE('a-b-c','-','+') | a+b+c |
    | INSTR(str, sub) | 部分文字列位置 | INSTR('hello','e') | 2 |
    | LOCATE(sub, str) | 部分文字列位置 | LOCATE('e','hello') | 2 |
    | LPAD(str, len, pad) | 左埋め | LPAD('5',3,'0') | 005 |
    | RPAD(str, len, pad) | 右埋め | RPAD('5',3,'0') | 500 |
    | REVERSE(str) | 逆順 | REVERSE('abc') | cba |
    | STRCMP(str1, str2) | 文字列比較 | STRCMP('a','b') | -1 |
    | FIND_IN_SET(str, list) | 集合内位置 | FIND_IN_SET('b','a,b,c') | 2 |

### 2. 抽出系関数

    | 関数 | 基本構文 | 意味 | ルール | 例 | 結果 |
    |----|----|----|----|----|----|
    | LEFT | LEFT(str, n) | 左から n 文字取得 | n ≤ 0 は空文字 | LEFT('abcdef', 3) | abc |
    | RIGHT | RIGHT(str, n) | 右から n 文字取得 | n ≤ 0 は空文字 | RIGHT('abcdef', 2) | ef |
    | SUBSTRING | SUBSTRING(str, pos, len) | pos から len 文字取得 | pos は 1 始まり | SUBSTRING('abcdef', 2, 3) | bcd |
    | SUBSTRING | SUBSTRING(str, pos) | pos から末尾まで | pos は負数可 | SUBSTRING('abcdef', 3) | cdef |
    | SUBSTRING | SUBSTRING(str, -n) | 右から n 文字取得 | RIGHT と同等 | SUBSTRING('abcdef', -2) | ef |
    | MID | MID(str, pos, len) | SUBSTRING の別名 | 完全同等 | MID('abcdef', 2, 2) | bc |
    | SUBSTR | SUBSTR(str, pos, len) | SUBSTRING の別名 | 完全同等 | SUBSTR('abcdef', 1, 4) | abcd |
    | CHAR_LENGTH | CHAR_LENGTH(str) | 文字数 | 中国語1文字=1 | CHAR_LENGTH('你好a') | 3 |
    | LENGTH | LENGTH(str) | バイト数 | UTF8中国語=3バイト | LENGTH('你好a') | 7 |

### 3. 日付関数
    | 関数 | 構文 | 意味 | ルール | 例 | 結果 |
    |----|----|----|----|----|----|
    | CURDATE | CURDATE() | 現在日付 | 時刻なし | CURDATE() | 2026-01-21 |
    | CURRENT_DATE | CURRENT_DATE | 現在日付 | CURDATE 同等 | CURRENT_DATE | 2026-01-21 |
    | NOW | NOW() | 現在日時 | 時分秒あり | NOW() | 2026-01-21 14:xx:xx |
    | SYSDATE | SYSDATE() | システム時刻 | トランザクション非同期 | SYSDATE() | 2026-01-21 14:xx:xx |
    | CURRENT_TIMESTAMP | CURRENT_TIMESTAMP | 現在タイムスタンプ | NOW 同等 | CURRENT_TIMESTAMP | 2026-01-21 14:xx:xx |
    | DATE | DATE(expr) | 日付部分抽出 | 時刻除去 | DATE('2026-01-21 10:30:00') | 2026-01-21 |
    | TIME | TIME(expr) | 時刻部分抽出 | 日付除去 | TIME('2026-01-21 10:30:00') | 10:30:00 |
    | YEAR | YEAR(date) | 年取得 | 整数返却 | YEAR('2026-01-21') | 2026 |
    | MONTH | MONTH(date) | 月取得 | 1–12 | MONTH('2026-01-21') | 1 |
    | DAY | DAY(date) | 日取得 | 1–31 | DAY('2026-01-21') | 21 |
    | HOUR | HOUR(time) | 時取得 | 0–23 | HOUR('14:30:00') | 14 |
    | MINUTE | MINUTE(time) | 分取得 | 0–59 | MINUTE('14:30:45') | 30 |
    | SECOND | SECOND(time) | 秒取得 | 0–59 | SECOND('14:30:45') | 45 |
    | DAYOFWEEK | DAYOFWEEK(date) | 曜日番号 | 日曜=1 | DAYOFWEEK('2026-01-21') | 4 |
    | WEEKDAY | WEEKDAY(date) | 曜日番号 | 月曜=0 | WEEKDAY('2026-01-21') | 2 |
    | DAYOFMONTH | DAYOFMONTH(date) | 月内日数 | DAYと同等 | DAYOFMONTH('2026-01-21') | 21 |
    | DAYOFYEAR | DAYOFYEAR(date) | 年内日数 | 1–366 | DAYOFYEAR('2026-01-21') | 21 |
    | LAST_DAY | LAST_DAY(date) | 月末日 | date返却 | LAST_DAY('2026-02-01') | 2026-02-28 |
    | DATE_ADD | DATE_ADD(date, INTERVAL n unit) | 日付加算 | 複数単位対応 | DATE_ADD('2026-01-21', INTERVAL 3 DAY) | 2026-01-24 |
    | DATE_SUB | DATE_SUB(date, INTERVAL n unit) | 日付減算 | 同上 | DATE_SUB('2026-01-21', INTERVAL 1 MONTH) | 2025-12-21 |
    | DATEDIFF | DATEDIFF(d1, d2) | 日付差 | d1 - d2（日） | DATEDIFF('2026-01-21','2026-01-01') | 20 |
    | TIMESTAMPDIFF | TIMESTAMPDIFF(unit, t1, t2) | 時間差 | 単位精度 | TIMESTAMPDIFF(DAY,'2026-01-01','2026-01-21') | 20 |
    | STR_TO_DATE | STR_TO_DATE(str, fmt) | 文字列→日付 | 書式必要 | STR_TO_DATE('2026-01-21','%Y-%m-%d') | 2026-01-21 |
    | DATE_FORMAT | DATE_FORMAT(date, fmt) | 日付→文字列 | 書式化 | DATE_FORMAT('2026-01-21','%Y/%m/%d') | 2026/01/21 |

### 4. フロー制御関数
    | 関数 | 役割 | 構文 | 例 | 結果 |
    |---|---|---|---|---|
    | IF(expr, v1, v2) | 条件判定 | IF(条件, 真値, 偽値) | IF(60>=60,'合格','不合格') | 合格 |
    | IFNULL(v1, v2) | v1がNULLならv2 | IFNULL(v1, v2) | IFNULL(NULL,0) | 0 |
    | NULLIF(v1, v2) | v1=v2ならNULL | NULLIF(v1,v2) | NULLIF(5,5) | NULL |
    | CASE WHEN | 複数条件分岐 | CASE WHEN 条件 THEN 値 END | CASE WHEN score>=60 THEN '合格' END | 合格 |
    | CASE expr WHEN | 等値分岐 | CASE expr WHEN 値 THEN 値 END | CASE sex WHEN '男' THEN 1 END | 1 |

# 七、制約
### 1. 制約分類
    | 制約 | 役割 | 位置 | 例 | 補足 |
    |---|---|---|---|---|
    | PRIMARY KEY | 主キー（一意+非NULL） | 列/表 | id INT PRIMARY KEY | 1テーブル1つ |
    | UNIQUE | 一意制約 | 列/表 | email VARCHAR(50) UNIQUE | 複数設定可 |
    | NOT NULL | 非NULL制約 | 列 | name VARCHAR(20) NOT NULL | NULL禁止 |
    | DEFAULT | デフォルト値 | 列 | status INT DEFAULT 1 | INSERT時省略可 |
    | CHECK | 条件制約 | 列/表 | age INT CHECK (age >= 0) | MySQL 8+有効 |
    | FOREIGN KEY | 外部キー | 表 | FOREIGN KEY (cid) REFERENCES class(id) | 表関連付け |
    | AUTO_INCREMENT | 自動採番 | 列 | id INT AUTO_INCREMENT | 主キー併用多い |

### 2. 主キーと外部キー
- テーブル作成後に主キー追加
    ```sql
    ALTER TABLE table_name
    ADD PRIMARY KEY (pk_column);
    -- 主キーは UNIQUE + NOT NULL を含む
    ```
- 主キー削除
    ```sql
    ALTER TABLE table_name
    DROP PRIMARY KEY;
    ```
- 主キー情報確認
    ```sql
    SHOW KEYS FROM table_name WHERE Key_name = 'PRIMARY';
    ```
- テーブル作成時に外部キー定義
    ```sql
    CREATE TABLE table_name (
    ...
    ...
    ...

    CONSTRAINT fk_name
    FOREIGN KEY (fk_column) REFERENCES parent_table(parent_column)
    );
    -- CONSTRAINT名は省略可。ただし削除時は名前が必要。
    ```
- テーブル作成後に外部キー追加
    ```sql
    ALTER TABLE table_name
    ADD CONSTRAINT fk_name
    FOREIGN KEY (fk_column) REFERENCES parent_table(parent_column)
    ```

### 3. 外部キー制約
- RESTRICT / NO ACTION（デフォルト）
    ```sql
    ALTER TABLE table_name ADD CONSTRAINT fk_name
    FOREIGN KEY (fk_column) REFERENCES parent_table(parent_column)
    ON UPDATE RESTRICT ON DELETE RESTRICT
    -- NO ACTION は RESTRICT と同じ。
    -- 子テーブル参照がある場合、親テーブルの更新/削除は不可。
    ```
- CASCADE
    ```sql
    ON UPDATE CASCADE
    ON DELETE CASCADE
    -- 親更新/削除時に子テーブルも更新/削除
    ```
- SET NULL
    ```sql
    ON UPDATE SET NULL
    ON DELETE SET NULL
    -- 親更新/削除時に子側FKをNULLへ（FK列がNULL許可必要）
    ```
- SET DEFAULT: MySQLでは使用不可

### 4. 外部キー制約変更フロー
- 1. 外部キー名確認
    ```sql
    SHOW CREATE TABLE orders;
    ```
- 2. 既存外部キー削除
    ```sql
    ALTER TABLE table_name
    DROP FOREIGN KEY fk_name;
    ```
- 3. 外部キー再追加
    ```sql
    ALTER TABLE table_name
    ADD CONSTRAINT fk_name
    FOREIGN KEY (fk_column) REFERENCES parent_table(parent_column)
    ON DELETE SET NULL ON UPDATE SET NULL;
    ```

# 八、多表検索
### 1. 内部結合
- 暗黙的内部結合
    ```sql
    SELECT columns FROM table1, table2
    WHERE condition;
    -- 列参照は table.column の形を推奨
    ```
- 明示的内部結合
    ```sql
    SELECT columns FROM table1
    INNER JOIN table2 ON join_condition;
    -- INNER は省略可
    ```
* 内部結合ではテーブル別名を使うと見やすい。別名設定後は元テーブル名では参照しない。
* 列別名を付けても、元列名の意味は保持される。

### 2. 外部結合
- 左外部結合
    ```sql
    SELECT columns FROM table1
    LEFT OUTER JOIN table2 ON condition;
    -- OUTER は省略可
    -- table1全件 + 両表の一致分。不一致側はNULL補完。
    ```
- 右外部結合
    ```sql
    SELECT columns FROM table1
    RIGHT JOIN table2 ON condition;
    -- table2全件 + 両表の一致分
    ```

### 3. 自己結合
    ```sql
    SELECT columns FROM table1 alias1
    JOIN table1 alias2 ON condition;
    -- 自己結合では別名必須
    -- INNER / LEFT / RIGHT いずれも可
    ```

### 4. UNION検索
- 複数検索結果の結合
    ```sql
    SELECT columns FROM ...
    UNION ALL
    SELECT columns FROM ...
    -- UNION ALL は重複保持、UNION は重複排除
    -- 2つのSELECTは列数一致、型互換が必要
    -- ORDER BY は最後に1回のみ
    ```

### 5. サブクエリ
- 入れ子検索
- スカラサブクエリ
    > 単一値（数値/文字/時刻など）を返す。=, <>, >, >=, <, <= と併用。
- 列サブクエリ
    > 1列を返す。IN, NOT IN, ANY, SOME, ALL と併用。
    - **IN**：集合に含まれる
    - **NOT IN**：集合に含まれない
    - **ANY / SOME**：いずれか満たせばよい
    - **ALL**：すべて満たす必要あり
    - **EXIST**：存在すれば真
- 行サブクエリ
    > 1行を返す。=, <>, IN, NOT IN と併用。

# 九、トランザクション
    トランザクション（Transaction）は複数SQL操作を一つの不可分な作業単位として扱う仕組み。すべて成功するか、すべて失敗するかのどちらかになる。

- 自動コミット設定確認
    ```sql
    SELECT @@autocommit;
    ```
- コミット方式設定
    ```sql
    SET @@autocommit = 0;
    -- 1: 自動コミット, 0: 手動コミット
    ```
- コミット
    ```sql
    COMMIT;
    ```
- ロールバック
    ```sql
    ROLLBACK;
    ```
- トランザクション開始
    ```sql
    START / BEGIN TRANSACTION;
    ```

# 待更新...
