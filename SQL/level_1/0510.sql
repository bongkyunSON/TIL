CREATE DATABASE pokemon;

USE pokemon;
CREATE TABLE mypokemon(
				number INT,
                name VARCHAR(20),
                type VARCHAR(10)
                );
                

CREATE TABLE pokemon.mypokemon(
				number INT,
                name VARCHAR(20),
                type VARCHAR(10)
                );
                
                
INSERT INTO mypokemon(number, name, type)
VALUES(10, 'caterpie', 'bug'),
	  (25, 'pikachu', 'electric'),
      (133, 'eevee', 'nomal');
      
      
SELECT * FROM mypokemon;

USE pokemon;
CREATE TABLE mynewpokemon(
				number INT,
                name VARCHAR(20),
                type VARCHAR(10)
                );
                
INSERT INTO mynewpokemon (number, name, type)
VALUES (77, '포니타', '불꽃'),
		(132, '메타몽', '노멀'),
        (151, '뮤', '에스퍼');
        
        
	

SELECT * FROM mynewpokemon;


USE pokemon;

SELECT * FROM mypokemon;

ALTER TABLE mypokemon
RENAME myoldpokemon;

SELECT * FROM myoldpokemon;

ALTER TABLE myoldpokemon
CHANGE COLUMN name eng_nm VARCHAR(20);

SELECT * FROM myoldpokemon;


ALTER TABLE mynewpokemon
CHANGE COLUMN name kor_nm VARCHAR(10);

SELECT * FROM mynewpokemon;

USE pokemon;
TRUNCATE TABLE myoldpokemon;
SELECT * FROM myoldpokemon;

SELECT * FROM mynewpokemon;
DROP TABLE mynewpokemon;

SELECT 123*456;

SELECT 2310/30;

USE pokemon;
SELECT 'pikachu' as pokemon;

SELECT * FROM pokemon;



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


SELECT 123*456;

SELECT 2310 / 30;

SELECT '피카츄' AS '포케몬';

USE pokemon;
SELECT * FROM mypokemon;

SELECT name
FROM mypokemon;

SELECT name, height, weight
FROM mypokemon;

SELECT DISTINCT height
FROM mypokemon;

SELECT name, attack*2 AS attack2, attack
FROM mypokemon; 

SELECT name AS 이름
FROM mypokemon;

SELECT attack AS 공격력, defense AS 방어력
FROM mypokemon;

SELECT height * 100 AS 'height(cm)'
FROM mypokemon;

SELECT *
FROM mypokemon
LIMIT 1;


SELECT name AS 영문명, height AS '키(m)', weight AS '몸무게(kg)'
FROM mypokemon
LIMIT 2;

SELECT name, attack + defense + speed AS total
FROM mypokemon;

SELECT name, weight / height^2 AS BMI
FROM mypokemon;


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

SELECT name, type
FROM mypokemon
WHERE type IN('normal', 'fire', 'water', 'grass');


SELECT name, attack
FROM mypokemon
WHERE attack BETWEEN 40 AND 60;


SELECT name
FROM mypokemon
WHERE name LIKE '%e%';


SELECT *
FROM mypokemon
WHERE name LIKE '%i%' AND speed >= 50;

SELECT *
FROM mypokemon
WHERE attack BETWEEN 40 AND 60;

USE pokemon;

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
WHERE  attack + defense + speed >= 150;

USE pokemon;








