# 一、 语法类型与数据类型

### 1. SQL语法与分类
| 分类 | 全称 | 作用 | 常见语法 |
|---|---|---|---|
| DDL | Data Definition Language | 定义、修改数据库与表结构 | CREATE、ALTER、DROP、TRUNCATE |
| DML | Data Manipulation Language | 对表中数据进行增删改 | INSERT、UPDATE、DELETE |
| DQL | Data Query Language | 查询数据 | SELECT、FROM、WHERE、GROUP BY、HAVING、ORDER BY、LIMIT |
| DCL | Data Control Language | 权限与用户控制 | GRANT、REVOKE |
| TCL | Transaction Control Language | 事务控制 | BEGIN / START TRANSACTION、COMMIT、ROLLBACK |

### 2. SQL数据类型
| 分类 | 类型 | 字节 | 范围 | 常见用途 |
|---|---|---|---|---|
| 数值型 | TINYINT | 1B | -128 ~ 127 | 状态位 |
| 数值型 | TINYINT UNSIGNED | 1B | 0 ~ 255 | 计数、小ID |
| 数值型 | SMALLINT | 2B | -32,768 ~ 32,767 | 小范围数值 |
| 数值型 | SMALLINT UNSIGNED | 2B | 0 ~ 65,535 | 年龄、数量 |
| 数值型 | INT / INTEGER | 4B | $-2^{31} \sim 2^{31}-1$ | 普通整型 |
| 数值型 | INT UNSIGNED | 4B | $0 \sim 2^{32}-1$ | 主键ID |
| 数值型 | BIGINT | 8B | $-2^{63} \sim 2^{63}-1$ | 大整数 |
| 数值型 | BIGINT UNSIGNED | 8B | $0 \sim 2^{64}-1$ | 用户ID、订单ID |
| 数值型 | FLOAT | 4B | 约 ±3.4E38 | 概率、评分 |
| 数值型 | DOUBLE | 8B | 约 ±1.7E308 | 计算结果 |
| 数值型 | DECIMAL(p,s) | 按精度 | 精确表示 | 金额 |
| 字符串 | CHAR(n) | n 字符 | 固定长度 | 固定编码 |
| 字符串 | VARCHAR(n) | ≤ n 字符 | 变长 | 姓名、账号 |
| 字符串 | TEXT | ≤ 65,535 字符 | 长文本 | 备注 |
| 时间 | DATE | 3B | 1000-01-01 ~ 9999-12-31 | 日期 |
| 时间 | TIME | 3B | -838:59:59 ~ 838:59:59 | 时间 |
| 时间 | DATETIME | 8B | 1000-01-01 ~ 9999-12-31 | 业务时间 |
| 时间 | TIMESTAMP | 4B | 1970-01-01 ~ 2038-01-19 | 创建/更新时间 |
| 布尔 | BOOLEAN / BOOL | 1B | 0 / 1 | 是否有效 |
| 枚举 | ENUM | 1–2B | 预定义值 | 状态 |


# 二、 DDL
### 1. 数据库操作
- 查询所有数据库 (DQL)
    ```sql
    SHOW DATABASES;
    ```
- 查询当前数据库 (DQL)
    ```sql
    SELECT DATABASE();
    ```
- 创建数据库
    ```sql
    CREATE DATABASE 表名;
    ```
    ```sql
    CREATE DATABASE (IF NOT EXISTS) 表名 (DEFAULT CHARSET 字符集) (COLLATE 排序);
    --加 IF NOT EXISTS避免报错；后面可以选字符集和排序规则。
    ```
### 2. 表操作

- 查询当前数据库所有表 (DQL)
    ```sql
    SHOW TABLES;
    ```
- 查询表结构 (DQL)
    ```sql
    DESC 表名;
    ```
- 查询建表语句 (DQL)
    ```sql
    SHOW CREATE TABLE 表名;
    ```
- 表创建
    ```sql
    CREATE TABLE 表名(
        字段1 类型，
        字段2 类型 
    )
    ```
    ```sql
    CREATE TABLE 表名(
        字段1 类型 COMMENT '字段注释'，
        字段2 类型 COMMENT '字段注释'
    ) COMMENT = '表注释'
    ```
- 表删除
    ```sql
    DROP TABLE 表名;
    --DROP TABLE IF EXISTS 表名避免报错
    --DROP TABLE table1,table2,table3
    ```
- 清空表数据
    ```sql
    TRUNCATE TABLE 表名
    --不能IF EXISTS，属性不变
    ```
- 修改表名
    ```sql
    ALTER TABLE 旧表名 RENAME TO 新表名;
    --属性不会变
    ```
### 3. 表字段操作
- 为表添加字段
    ```sql
    ALTER TABLE 表名 ADD 字段 类型
    --后面可以加 COMMENT 注释 约束
    ```
- 删除字段
    ```sql
    ALTER TABLE 表名 DROP 字段;
    --其他SQL里也写成ALTER TABLE 表名 DROP COLUMN 字段;
    ```
- 修改数据类型
    ```sql
    ALTER TABLE 表名 
    MODIFY 字段名 新类型;
    --会覆盖所有属性，可追加 NOT NULL/ DEFAULT/ COMMENT
    ```
- 修改字段名和类型
    ```sql
    ALTER TABLE 表名 
    CHANGE 旧字段 新字段 新类型;
    --Change和modify一样也会丢失属性
    ```
# 三、DML
### 1. 添加数据
- 给指定字段添加数据
    ```sql
    INSERT INTO 表名 (字段1, 字段2)
    VALUES (值1, 值2);
    -- 字符串和日期的值需要 ' '
    ```
- 给全部字段添加数据
    ```sql
    INSERT INTO 表名
    VALUES (值1，值2，值3);
    ```
- 批量添加数据
    ```sql
    INSERT INTO 表名 (字段1, 字段2)
    VALUES (值1, 值2),
    (值1, 值2),
    (值1, 值2);
    ```
    ```sql
    INSERT INTO 表名
    VALUES (值1, 值2, 值3), 
    (值1, 值2, 值3),
    (值1, 值2, 值3);
    ```
### 2. 修改数据
- 修改全表数据
    ```sql
    UPDATE 表名 SET 字段1=值1, 字段2=值2;
    ```
- 修改特定数据
    ```sql
    UPDATE 表名 字段1=值1, 字段2=值2 WHERE 条件;
    ```
- 删除数据
    ```sql
    DELETE FROM 表名 WHERE 条件;
    --不写where则删除全部
    ```
# 四、DQL
### 1. DQL 语法
- 
    ```sql
    SELECT      字段
    FROM        表名
    WHERE       条件
    GROUP BY    分组字段   
    HAVING      分组后条件
    ORDER BY    排序字段
    LIMIT       分页参数
    ```
### 2. DQL基本查询
- 查询多字段
    ```sql
    SELECT 字段1, 字段2, 字段3 FROM 表名;
    ```
    ```sql
    SELECT * FROM 表名;
    ```
- 设置字段别名
    ```sql
    SELECT 字段1 AS 别名, 字段2 AS 别名 FROM 表名;
    -- AS可以省略，SELECT 字段 别名 FROM 表名;
    ```
- 去重
    ```sql
    SELECT DISTINCT 字段 FROM 表名;
    -- SELECT DISTINCT 字段1，字段2 FROM 表名;
    --则为查询双字段的唯一组合(会更多)
    ```
### 3. 条件查询
- 语法
    ```sql
    SELECT 字段 FROM 表名 WHERE 条件;
    ```
- 条件
    | 分类 | 条件 | 含义 | 示例 |
    |---|---|---|---|
    | 比较 | = | 等于 | price = 100 |
    | 比较 | != / <> | 不等于 | price != 100 |
    | 比较 | > | 大于 | price > 100 |
    | 比较 | < | 小于 | price < 100 |
    | 比较 | >= | 大于等于 | price >= 100 |
    | 比较 | <= | 小于等于 | price <= 100 |
    | 范围 | BETWEEN a AND b | 区间内（含左右边界） | price BETWEEN 100 AND 200 |
    | 范围 | NOT BETWEEN | 不在区间内 | price NOT BETWEEN 100 AND 200 |
    | 集合 | IN (...) | 在集合中 | brand IN ('BMW','Audi') |
    | 集合 | NOT IN (...) | 不在集合中 | brand NOT IN ('BMW','Audi') |
    | 空值 | IS NULL | 是空值 | discount IS NULL |
    | 空值 | IS NOT NULL | 非空值 | discount IS NOT NULL |
    | 模糊 | LIKE '%x%' | 包含 | model LIKE '%X%' |
    | 模糊 | LIKE 'x%' | 以 x 开头 | model LIKE 'X%' |
    | 模糊 | LIKE '%x' | 以 x 结尾 | model LIKE '%X' |
    | 模糊 | NOT LIKE | 不匹配 | model NOT LIKE '%X%' |

- 逻辑运算符

    | 运算符 | 含义 | 说明 | 示例 |
    |---|---|---|---|
    | AND / && | 且 | 所有条件都为真 | price > 100 AND brand = 'BMW' |
    | OR / \|  \|| 或 | 任一条件为真 | brand = 'BMW' OR brand = 'Audi' |
    | NOT / !| 非 / 取反 | 条件取反 | NOT price > 100 |
    | XOR | 异或 | 仅一个条件为真 | brand = 'BMW' XOR brand = 'Audi' |

- 聚合函数 
    | 函数 | 含义 | 作用对象 | 示例 | 说明 |
    |---|---|---|---|---|
    | COUNT(*) | 统计行数 | 行 | COUNT(*) | 包含 NULL |
    | COUNT(col) | 统计非空值数量 | 列 | COUNT(price) | 忽略 NULL |
    | SUM(col) | 求和 | 数值列 | SUM(price) | 忽略 NULL |
    | AVG(col) | 平均值 | 数值列 | AVG(price) | 忽略 NULL |
    | MAX(col) | 最大值 | 列 | MAX(price) | 可用于字符串 |
    | MIN(col) | 最小值 | 列 | MIN(price) | 可用于字符串 |
### 4. 分组查询
- 语法
    ```sql
    SELECT 字段 FROM 表名 (WHERE 条件) GROUP BY 分组字段名 (HAVING 分组后过滤条件)
    ```
    where条件是分组之前过滤，having是分组之后对结果过滤；
    where不能用聚合函数，having可以。
    执行顺序 where > 聚合函数 > having
    分组后一般只查询分组字段和聚合函数
### 5. 排序和分页查询
- 排序查询
    ```sql
    SELECT 字段 FROM 表名 
    ORDER BY 字段1 ASC, 字段2 DESC;
    -- asc为升序(默认可省略)，desc为降序。
    ```
- 分页查询
    ```sql
    SELECT 字段 FROM 表名 LIMIT 起始索引, 查询记录数;
    ```
    起始索引 = 每页记录数 * (查询页码 - 1)
    第一页数据可省略，写为limit 10

### 6. DQL语句的执行顺序
- 
    ```sql
    FROM - WHERE - GROUP BY - HAVING - SELECT - ORDER BY - LIMIT 
    ```
# 五、DCL
### 1. 管理用户
- 查询用户
    ```sql
    USE mysql;
    SELECT * FROM user;
    -- 用户信息在系统数据库mysql的user表中
    ```
- 创建用户
    ```sql
    CREATE USER '用户名'@'主机名' IDENTIFIED BY '设定密码';
    -- 本机的主机名为localhost
    -- 分配任意主机权限，用 '%' 作为主机名
    ```
- 删除用户
    ```sql
    DROP USER '用户名@主机名';
    ```
- 修改用户密码
    ```sql
    ALTER USER '用户名@主机名' 
    IDENTIFIED WITH caching_sha2_password BY '新密码';
    -- mysql8.0以前不用caching_sha2_password，用 mysql_native_password
    ```
### 2. 用户权限控制
- 查询权限
    ```sql
    SHOW GRANTS FOR '用户名'@'主机名';
    ```
- 授予权限
    ```sql
    GRANT 权限 ON 数据库名.表名 TO '用户名'@'主机名';
    -- 所有权限可以写ALL PRIVILEGES,PRIVILEGES 可省略
    -- 所有数据库/表可以写*.*
    ```
- 撤销权限
    ```sql
    REVOKE 权限 ON 数据库名.表名 TO '用户名'@'主机名';
    ```
- 权限列表
    | 权限名 | 作用说明 | 常见使用场景 |
    |------|---------|-------------|
    | ALL / ALL PRIVILEGES | 所有权限 | 管理员 / root |
    | USAGE | 无任何权限（仅表示账号存在） | 默认状态 |
    | SELECT | 查询数据 | 只读用户 |
    | INSERT | 插入数据 | 录入数据 |
    | UPDATE | 更新数据 | 修改数据 |
    | DELETE | 删除数据 | 删除记录 |
    | CREATE | 创建数据库 / 表 | 建表 |
    | DROP | 删除数据库 / 表 | 删表 / 删库 |
    | ALTER | 修改表结构 | 改字段 |
    | INDEX | 创建 / 删除索引 | 性能优化 |
    | CREATE VIEW | 创建视图 | 报表 |
    | SHOW VIEW | 查看视图定义 | 运维 / 查看 |
    | TRIGGER | 创建触发器 | 高级逻辑 |
    | EXECUTE | 执行存储过程 | 存储过程 |
    | EVENT | 创建事件调度器 | 定时任务 |
    | REFERENCES | 外键约束 | 表关联 |
    | GRANT OPTION | 允许把自己的权限再授权给别人 | 管理权限 |
# 六、函数
### 1. 字符串函数
- 
    | 函数名 | 作用 | 示例 | 结果 |
    |------|------|------|------|
    | LENGTH(str) | 返回字符串字节长度 | LENGTH('abc') | 3 |
    | CHAR_LENGTH(str) | 返回字符个数 | CHAR_LENGTH('你好') | 2 |
    | CONCAT(str1, str2, ...) | 拼接字符串 | CONCAT('a','b','c') | abc |
    | CONCAT_WS(sep, str1, ...) | 用分隔符拼接 | CONCAT_WS('-', '2024','01','01') | 2024-01-01 |
    | UPPER(str) | 转为大写 | UPPER('abc') | ABC |
    | LOWER(str) | 转为小写 | LOWER('ABC') | abc |
    | LEFT(str, n) | 从左截取 n 个字符 | LEFT('abcdef',3) | abc |
    | RIGHT(str, n) | 从右截取 n 个字符 | RIGHT('abcdef',3) | def |
    | SUBSTRING(str, start, len) | 截取子串 | SUBSTRING('abcdef',2,3) | bcd |
    | SUBSTR(str, start, len) | SUBSTRING 的别名 | SUBSTR('abcdef',2,3) | bcd |
    | TRIM(str) | 去除两端空格 | TRIM('  hi  ') | hi |
    | LTRIM(str) | 去除左侧空格 | LTRIM('  hi') | hi |
    | RTRIM(str) | 去除右侧空格 | RTRIM('hi  ') | hi |
    | REPLACE(str, from, to) | 字符串替换 | REPLACE('a-b-c','-','+') | a+b+c |
    | INSTR(str, sub) | 返回子串位置 | INSTR('hello','e') | 2 |
    | LOCATE(sub, str) | 返回子串位置 | LOCATE('e','hello') | 2 |
    | LPAD(str, len, pad) | 左填充到指定长度 | LPAD('5',3,'0') | 005 |
    | RPAD(str, len, pad) | 右填充到指定长度 | RPAD('5',3,'0') | 500 |
    | REVERSE(str) | 字符串反转 | REVERSE('abc') | cba |
    | STRCMP(str1, str2) | 比较字符串 | STRCMP('a','b') | -1 |
    | FIND_IN_SET(str, list) | 查找在集合中的位置 | FIND_IN_SET('b','a,b,c') | 2 |

2. 取值函数
- 
    | 函数 | 基本语法 | 含义 | 关键规则 | 示例 | 结果 |
    |----|----|----|----|----|----|
    | LEFT | LEFT(str, n) | 从左取 n 个字符 | n ≤ 0 返回空串 | LEFT('abcdef', 3) | abc |
    | RIGHT | RIGHT(str, n) | 从右取 n 个字符 | n ≤ 0 返回空串 | RIGHT('abcdef', 2) | ef |
    | SUBSTRING | SUBSTRING(str, pos, len) | 从 pos 开始取 len 个 | pos 从 1 开始 | SUBSTRING('abcdef', 2, 3) | bcd |
    | SUBSTRING | SUBSTRING(str, pos) | 从 pos 一直取到末尾 | pos 可为负 | SUBSTRING('abcdef', 3) | cdef |
    | SUBSTRING | SUBSTRING(str, -n) | 从右往左取 n 个 | 等价 RIGHT | SUBSTRING('abcdef', -2) | ef |
    | MID | MID(str, pos, len) | SUBSTRING 的别名 | 完全等价 | MID('abcdef', 2, 2) | bc |
    | SUBSTR | SUBSTR(str, pos, len) | SUBSTRING 的别名 | 完全等价 | SUBSTR('abcdef', 1, 4) | abcd |
    | CHAR_LENGTH | CHAR_LENGTH(str) | 字符长度 | 中文算 1 | CHAR_LENGTH('你好a') | 3 |
    | LENGTH | LENGTH(str) | 字节长度 | UTF8 中文算 3 | LENGTH('你好a') | 7 |
### 3. 日期函数
- 
    | 函数 | 语法 | 含义 | 关键规则 | 示例 | 结果 |
    |----|----|----|----|----|----|
    | CURDATE | CURDATE() | 当前日期 | 不含时间 | CURDATE() | 2026-01-21 |
    | CURRENT_DATE | CURRENT_DATE | 当前日期 | 同 CURDATE | CURRENT_DATE | 2026-01-21 |
    | NOW | NOW() | 当前日期+时间 | 含时分秒 | NOW() | 2026-01-21 14:xx:xx |
    | SYSDATE | SYSDATE() | 系统当前时间 | 非事务一致 | SYSDATE() | 2026-01-21 14:xx:xx |
    | CURRENT_TIMESTAMP | CURRENT_TIMESTAMP | 当前时间戳 | 同 NOW | CURRENT_TIMESTAMP | 2026-01-21 14:xx:xx |
    | DATE | DATE(expr) | 取日期部分 | 去掉时间 | DATE('2026-01-21 10:30:00') | 2026-01-21 |
    | TIME | TIME(expr) | 取时间部分 | 去掉日期 | TIME('2026-01-21 10:30:00') | 10:30:00 |
    | YEAR | YEAR(date) | 取年 | 返回整数 | YEAR('2026-01-21') | 2026 |
    | MONTH | MONTH(date) | 取月 | 1–12 | MONTH('2026-01-21') | 1 |
    | DAY | DAY(date) | 取日 | 1–31 | DAY('2026-01-21') | 21 |
    | HOUR | HOUR(time) | 取小时 | 0–23 | HOUR('14:30:00') | 14 |
    | MINUTE | MINUTE(time) | 取分钟 | 0–59 | MINUTE('14:30:45') | 30 |
    | SECOND | SECOND(time) | 取秒 | 0–59 | SECOND('14:30:45') | 45 |
    | DAYOFWEEK | DAYOFWEEK(date) | 星期几 | 周日=1 | DAYOFWEEK('2026-01-21') | 4 |
    | WEEKDAY | WEEKDAY(date) | 星期几 | 周一=0 | WEEKDAY('2026-01-21') | 2 |
    | DAYOFMONTH | DAYOFMONTH(date) | 月中第几天 | 同 DAY | DAYOFMONTH('2026-01-21') | 21 |
    | DAYOFYEAR | DAYOFYEAR(date) | 年中第几天 | 1–366 | DAYOFYEAR('2026-01-21') | 21 |
    | LAST_DAY | LAST_DAY(date) | 月末日期 | 返回 date | LAST_DAY('2026-02-01') | 2026-02-28 |
    | DATE_ADD | DATE_ADD(date, INTERVAL n unit) | 日期加 | 支持多单位 | DATE_ADD('2026-01-21', INTERVAL 3 DAY) | 2026-01-24 |
    | DATE_SUB | DATE_SUB(date, INTERVAL n unit) | 日期减 | 同上 | DATE_SUB('2026-01-21', INTERVAL 1 MONTH) | 2025-12-21 |
    | DATEDIFF | DATEDIFF(d1, d2) | 日期差 | d1 - d2（天） | DATEDIFF('2026-01-21','2026-01-01') | 20 |
    | TIMESTAMPDIFF | TIMESTAMPDIFF(unit, t1, t2) | 时间差 | 精确单位 | TIMESTAMPDIFF(DAY,'2026-01-01','2026-01-21') | 20 |
    | STR_TO_DATE | STR_TO_DATE(str, fmt) | 字符串→日期 | 需格式 | STR_TO_DATE('2026-01-21','%Y-%m-%d') | 2026-01-21 |
    | DATE_FORMAT | DATE_FORMAT(date, fmt) | 日期→字符串 | 常用 | DATE_FORMAT('2026-01-21','%Y/%m/%d') | 2026/01/21 |
### 4. 流程控制函数
- 
    | 函数 | 作用 | 语法 | 示例 | 结果 |
    |---|---|---|---|---|
    | IF(expr, v1, v2) | 条件判断，真取 v1 否取 v2 | IF(条件, 真值, 假值) | IF(60>=60,'及格','不及格') | 及格 |
    | IFNULL(v1, v2) | v1 为 NULL 时返回 v2 | IFNULL(v1, v2) | IFNULL(NULL,0) | 0 |
    | NULLIF(v1, v2) | v1 = v2 返回 NULL | NULLIF(v1,v2) | NULLIF(5,5) | NULL |
    | CASE WHEN | 多条件判断 | CASE WHEN 条件 THEN 值 END | CASE WHEN score>=60 THEN '及格' END | 及格 |
    | CASE expr WHEN | 等值判断 | CASE expr WHEN 值 THEN 值 END | CASE sex WHEN '男' THEN 1 END | 1 |
# 七、约束
### 1. 约束分类
- 
    | 约束 | 作用 | 位置 | 示例 | 说明 |
    |---|---|---|---|---|
    | PRIMARY KEY | 主键，唯一且非空 | 列 / 表 | id INT PRIMARY KEY | 一张表只能一个 |
    | UNIQUE | 唯一约束 | 列 / 表 | email VARCHAR(50) UNIQUE | 可多个 |
    | NOT NULL | 非空约束 | 列 | name VARCHAR(20) NOT NULL | 禁止 NULL |
    | DEFAULT | 默认值 | 列 | status INT DEFAULT 1 | 插入可省略 |
    | CHECK | 条件约束 | 列 / 表 | age INT CHECK (age >= 0) | MySQL 8+ 生效 |
    | FOREIGN KEY | 外键 | 表 | FOREIGN KEY (cid) REFERENCES class(id) | 关联表 |
    | AUTO_INCREMENT | 自增 | 列 | id INT AUTO_INCREMENT | 常配主键 |

### 2. 主键与外键
- 建表后追加主键
    ```sql
    ALTER TABLE 表名
    ADD PRIMARY KEY (主键字段);
    -- 主键默认约束为 UNIQUE 和 NOT NULL
    ```
- 删除主键
    ```sql
    ALTER TABLE 表名
    DROP PRIMARY KEY;
    ```
- 查看主键信息
    ```sql
    SHOW KEYS FROM 表名 WHERE Key_name = 'PRIMARY';
    ```
- 建表时定义外键
    ```sql
    CREATE TABLE 表名 (
    ...
    ...
    ...
    
    CONSTRAINT 外键名(现取)
    FOREIGN KEY (外键字段) REFERENCES 主表名(键字段)
    );
    -- 可不写CONSTRAINT，不取名，但DROP时需要知道外键名
    ```
- 建表后添加外键
    ```sql
    ALTER TABLE 表名
    ADD CONSTRAINT 键名
    FOREIGN KEY (外键字段) REFERENCES 主表名(键字段)
    ```

### 3. 外键约束
- RESTRICT/ NO ACTION 默认约束
    ```sql
    ALTER TABLE 表名 ADD CONSTRAINT 键名(现取)
    FOREIGN KEY (外键字段) REFERENCES 主表名(主表字段)
    ON UPDATE RESTRICT ON DELETE RESTRICT
    -- NO ACTION与RESTRICT一样。
    --更新/删除父表的行，会检测子表是否有外键引用。若有，则无法操作。
    ```
- CASCADE
    ```sql
    ON UPDATE CASCADE 
    ON DELETE CASCADE
    --父表更新/删除行时，子表外键对应记录也更新/删除
    ```
- SET NULL
    ```sql
    ON UPDATE SET NULL 
    ON DELETE SET NULL
    --父表更新/删除行时，子表外键对应记录更改为NULL（外键允许取NULL）
    ```
- SET DEFAULT: MySQL用不了

### 4.修改外键约束流程
1) 查看外键名
    ```sql 
    SHOW CREATE TABLE orders;
    ```
2) 删除原外键
    ```
    ALTER TABLE 表名
    DROP FOREIGN KEY 外键名;
    ```
3) 重新添加外键
    ```sql
    ALTER TABLE 表名
    ADD CONSTRAINT 键名
    FOREIGN KEY (外键字段) REFERENCES 主表名(键字段)
    ON DELETE SET NULL ON UPDATE SET NULL;
    ```
# 八、多表查询
### 1. 内连接
- 隐式内连接
    ```sql
    SELECT 字段列表 FROM 表1，表2
    WHERE 条件;
    -- 字段列表前面必须写表明，例如Students.student_id
    ```
- 显式内连接
    ```sql
    SELECT 字段列表 FROM 表1
    INNER JOIN 表2 ON 连接条件;
    -- inner可省略
    ```
* **常给表取别名方便内连接，但取别名后不能再使用原名调用表**
* **字段取别名后仍可使用原名。因为表别名是给“数据源”改名，字段别名只是给“输出列”换个显示名**
### 2. 外连接
- 左外连接
    ```sql
    SELECT 字段列表 FROM 表1
    LEFT OUTER JOIN 表2 ON 条件;
    -- OUTER 可省略
    -- 查询表1 所有数据，以及1和2的交集，不重合的部分用null补
    ```
- 右外连接
    ```sql
    SELECT 字段列表 FROM 表1
    RIGHT JOIN 表2 ON 条件;
    -- 查询表2 所有数据，以及1和2的交集
    ```
### 3. 自连接
- 
    ```sql
    SELECT 字段列表 FROM 表1 别名1 
    JOIN 表1 别名2 ON 条件;
    -- 自连接必须写别名 
    -- 可inner join，也可left/right join
    ```
### 4. 联合查询
- 把多次查询的结果合并
    ```sql
    SELECT 字段列表 FROM ...
    UNION ALL
    SELECT 字段列表 FROM ...
    -- UNION ALL表示结果不去重，UNION表示去重
    -- 两个字段列表的列数必须一样，列的类型需要兼容
    -- 只能在末尾写一个ORDER BY
    ```
### 5. 子查询
* 嵌套查询
- 标量子查询
    > 返回的结果是单个值（num，char，time等），常用操作符为=, <>, >, >=, <, <=
- 列子查询
    > 返回结果为一列，常用操作符为IN, NOT IN, ANY, SOME, ALL
    - **IN**：属于集合里面
    - **NOT IN**：不属于该集合
    - **ANY / SOME**：满足一个就行
    - **ALL**：全都满足才行
    - **EXIST**: 有就通过
- 行子查询
    >返回结果为一行，常用操作符为=, <>, IN, NOT IN


# 九、事务
事务（Transaction）是一组 SQL 操作，所有操作为一个不可分割的整体工作单位，一起提交给系统或者撤销操作请求。操作要么全部成功，要么全部失败。

- 查看事务的自动提交方式
    ```sql
    SELECT @@autocommit;
    ```
- 设置事务提交方式
    ```sql
    SET @@autocommit = 0;
    -- 1是自动提交，0是手动提交
    ```
- 提交事务
    ```sql
    COMMIT;
    ```
- 事务回滚
    ```sql
    ROLLBACK;
    ```
- 提交事务
    ```sql
    START / BEGIN TRANSACTION;
    ```


