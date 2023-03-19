DROP DATABASE IF EXISTS pokemon;
CREATE DATABASE pokemon;
USE pokemon;
CREATE TABLE mypokemon (
	number  int,
       name	varchar(20),
       type	varchar(10),
       attack int,
       defense int
);
INSERT INTO mypokemon (number, name, type, attack, defense)
VALUES (10, 'caterpie', 'bug', 30, 35),
	   (25, 'pikachu', 'electric', 55, 40),
	   (26, 'raichu', 'electric', 90, 55),
      	  (125, 'electabuzz', 'electric', 83, 57),
	   (133, 'eevee', 'normal', 55, 50),
         (137, 'porygon', 'normal', 60, 70),
	   (152, 'chikoirita', 'grass', 49, 65),
         (153, 'bayleef', 'grass', 62, 80),
         (172, 'pichu', 'electric', 40, 15),
         (470, 'leafeon', 'grass', 110, 130);
         
         
         
 
 
 

SELECT name, IF(number < 150, 'old', 'new') AS age
FROM mypokemon;


SELECT name, IF(attack + defense < 100, 'weak', 'strong') AS ability
FROM mypokemon;


SELECT type, IF(AVG(attack) >= 60, True, False) AS is_strong_type
FROM mypokemon
GROUP BY type;


SELECT name, IF(attack > 100 AND defense > 100, True, False) AS ace
FROM mypokemon;


SELECT name, 
CASE
WHEN number < 100 THEN '<100'
WHEN number < 200 THEN '<200'
WHEN number < 500 THEN '<500'
END AS number_bin
FROM mypokemon;


SELECT name, number, attack,
CASE 
WHEN number >=150 AND attack >=50 THEN 'new_strong'
WHEN number >=150 AND attack <50 THEN 'new_weak'
WHEN number < 150 AND attack >=50 THEN 'old_strong'
WHEN number < 150 AND attack < 50 THEN 'old_weak'
END AS age_attack
FROM mypokemon;


SELECT type,
CASE
WHEN count(1) =1 THEN 'solo'
WHEN count(1) < 3 THEN 'minor'
ELSE 'major'
END AS count_by_type
FROM mypokemon
GROUP BY type;

 

SET GLOBAL log_bin_trust_function_creators = 1;



DELIMITER //

CREATE FUNCTION isStrong(attack INT, defense INT)
	RETURNS VARCHAR(20)

BEGIN 
	DECLARE a INT;
    DECLARE b INT;
    DECLARE isStrong VARCHAR(20);
    SET a = attack;
    SET b = defense;
    SELECT CASE
			WHEN a + b > 120 THEN 'very strong'
            WHEN a + b > 90 THEN 'strong'
            ELSE 'not strong'
            END INTO isStrong;
	RETURN isStrong;
END


//
DELIMITER ; 

SELECT name, isStrong(attack, defense) AS isStrong
FROM mypokemon;



















