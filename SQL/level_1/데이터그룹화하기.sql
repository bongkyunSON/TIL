DROP DATABASE IF EXISTS pokemon;
CREATE DATABASE pokemon;
USE pokemon;
CREATE TABLE mypokemon (
number int,
name varchar(20),
type varchar(10),
height float,
weight float
);
INSERT INTO mypokemon (number, name, type, height, weight)
VALUES (10, 'caterpie', 'bug', 0.3, 2.9),
(25, 'pikachu', 'electric', 0.4, 6),
(26, 'raichu', 'electric', 0.8, 30),
(125, 'electabuzz', 'electric', 1.1, 30),
(133, 'eevee', 'normal', 0.3, 6.5),
(137, 'porygon', 'normal', 0.8, 36.5),
(152, 'chikoirita', 'grass', 0.9, 6.4),
(153, 'bayleef', 'grass', 1.2, 15.8),
(172, 'pichu', 'electric', 0.3, 2),
(470, 'leafeon', 'grass', 1, 25.5);


SELECT * FROM mypokemon;

SELECT type, AVG(weight)
FROM mypokemon
WHERE LENGTH(name) > 5
GROUP BY type 
HAVING AVG(weight) >= 20
ORDER BY 2 DESC;

SELECT type, MIN(height), MAX(height)
FROM mypokemon
WHERE number < 200
GROUP BY type
HAVING MAX(weight) >= 10 AND MIN(weight) >= 2
ORDER BY 2 DESC, 3 DESC;



SELECT type, AVG(height)
FROM mypokemon
GROUP BY type;


SELECT type, AVG(weight)
FROM mypokemon
GROUP BY type;


SELECT type, AVG(height) AS avg_height, AVG(weight) AS avg_weight
FROM mypokemon
GROUP BY type;

SELECT type
FROM mypokemon
GROUP BY type
HAVING AVG(height) >= 0.5;


SELECT type
FROM mypokemon
GROUP BY type
HAVING AVG(weight) >= 20;


SELECT type, SUM(number) AS '번호의합'
FROM mypokemon
GROUP BY type;


SELECT type, count(1), COUNT(*)
FROM mypokemon
WHERE height >= 0.5
GROUP BY type;


SELECT type, MIN(height)
FROM mypokemon
GROUP BY type;


SELECT type, MAX(weight)
FROM mypokemon
GROUP BY type;


SELECT type
FROM mypokemon
GROUP BY type
HAVING MIN(height) > 0.5 AND MAX(weight) < 30;




































