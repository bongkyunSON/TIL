DROP DATABASE IF EXISTS pokemon;
CREATE DATABASE pokemon;
USE pokemon;
CREATE TABLE mypokemon (
number int,
name varchar(20),
type varchar(20),
height float,
weight float,
attack float,
defense float,
speed float
);
INSERT INTO mypokemon (number, name, type, height, weight, attack, defense, speed)
VALUES (10, 'caterpie', 'bug', 0.3, 2.9, 30, 35, 45),
(25, 'pikachu', 'electric', 0.4, 6, 55, 40, 90),
(26, 'raichu', 'electric', 0.8, 30, 90, 55, 110),
(133, 'eevee', 'normal', 0.3, 6.5, 55, 50, 55),
(152, 'chikoirita', 'grass', 0.9, 6.4, 49, 65, 45);


SELECT type
FROM mypokemon
WHERE name = 'eevee';

SELECT attack, defense
FROM mypokemon
WHERE name = 'caterpie';

SELECT *
FROM mypokemon
WHERE weight > 6;

SELECT name
FROM mypokemon
WHERE height > 0.5 AND weight >= 6;

SELECT name AS weak_pokemon
FROM mypokemon
WHERE attack < 50 OR defense < 50;

SELECT *
FROM mypokemon
WHERE type != 'normal';

SELECT *
FROM mypokemon
WHERE NOT (type = 'normal');

SELECT name, type
FROM mypokemon
WHERE type IN ('normal', 'fire', 'water', 'grass');

SELECT name, attack
FROM mypokemon
WHERE attack BETWEEN 40 AND 60;

SELECT name, attack
FROM mypokemon
WHERE attack >= 40 AND attack <= 60;


SELECT name
FROM mypokemon
WHERE name LIKE '%e%';


SELECT *
FROM mypokemon
WHERE name LIKE '%i%' AND speed <= 50;


SELECT name, height, weight
FROM mypokemon
WHERE name LIKE '%chu';

SELECT name, defense
FROM mypokemon
WHERE name LIKE '%e' AND defense < 50;

SELECT name, attack, defense
FROM mypokemon
WHERE attack - defense >= 10 or defense - attack >= 10;

SELECT name, attack + defense + speed AS total
FROM mypokemon
WHERE attack + defense + speed >= 150;
