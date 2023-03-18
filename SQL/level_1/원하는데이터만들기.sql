DROP DATABASE IF EXISTS pokemon;
CREATE DATABASE pokemon;
USE pokemon;
CREATE TABLE mypokemon (
number INT,
name VARCHAR(20),
type VARCHAR(10),
attack INT,
defense INT,
capture_date DATE
);
INSERT INTO mypokemon (number, name, type, attack, defense, capture_date)
VALUES (10, 'caterpie', 'bug', 30, 35, '2019-10-14'),
(25, 'pikachu', 'electric', 55, 40, '2018-11-04'),
(26, 'raichu', 'electric', 90, 55, '2019-05-28'),
(125, 'electabuzz', 'electric', 83, 57, '2020-12-29'),
(133, 'eevee', 'normal', 55, 50, '2021-10-03'),
(137, 'porygon', 'normal', 60, 70, '2021-01-16'),
(152, 'chikoirita', 'grass', 49, 65, '2020-03-05'),
(153, 'bayleef', 'grass', 62, 80, '2022-01-01');





SELECT name, LENGTH(name)
FROM mypokemon
ORDER BY LENGTH(name);

SELECT name, RANK() OVER (ORDER BY defense DESC) AS defense_rank
FROM mypokemon;

SELECT name, DATEDIFF('2022-02-14', capture_date) AS days
FROM mypokemon;

SELECT name, DATEDIFF(capture_date, '2022-02-14') AS days
FROM mypokemon;




SELECT RIGHT(name, 3) AS last_char
FROM mypokemon;


SELECT LEFT(name, 2) AS left2
FROM mypokemon;


SELECT REPLACE(name, 'o', 'O') AS bigO
FROM mypokemon
WHERE name LIKE '%o%';

SELECT name, UPPER(CONCAT(LEFT(type, 1), RIGHT(type, 1))) AS type_code
FROM mypokemon;

SELECT DISTINCT UPPER(CONCAT(LEFT(type, 1), RIGHT(type, 1))) AS type_code
FROM mypokemon;

SELECT *
FROM mypokemon
WHERE LENGTH(name) > 8;

SELECT ROUND(AVG(attack)) AS avg_of_attack
FROM mypokemon;

SELECT ROUND(AVG(defense)) AS avg_of_defense
FROM mypokemon;

SELECT name, POWER(attack, 2) AS attack2
FROM mypokemon
WHERE LENGTH(name) < 8;


SELECT name, MOD(attack, 2) AS div2
FROM mypokemon;

SELECT name, ABS(attack - defense) AS diff
FROM mypokemon
WHERE attack <= 50;

SELECT CURRENT_DATE() AS now_date, CURRENT_TIME() AS now_time;

SELECT MONTH(capture_date) AS month_num, MONTHNAME(capture_date) AS month_eng
FROM mypokemon;

SELECT DAYOFWEEK(capture_date) AS day_num, DAYNAME(capture_date) AS day_eng
FROM mypokemon;

SELECT YEAR(capture_date) AS year, MONTH(capture_date) AS month, DAY(capture_date) AS day
FROM mypokemon;















