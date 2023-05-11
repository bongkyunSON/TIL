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

SELECT name, LENGTH(name)
FROM mypokemon
ORDER BY LENGTH(name);


DROP DATABASE IF EXISTS pokemon;
CREATE DATABASE pokemon;
USE pokemon;
CREATE TABLE mypokemon (
	 number INT,
       name	VARCHAR(20),
       type	VARCHAR(10),
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


SELECT name,
RANK() OVER(ORDER BY defense) AS defense_rank 
FROM mypokemon;


SELECT name,
DATEDIFF('2022-02-14', capture_date) AS days
FROM mypokemon
ORDER BY days;

SELECT RIGHT(name, 3) AS last_char
FROM mypokemon;


SELECT LEFT(name, 2) AS left2
FROM mypokemon;



SELECT name, REPLACE(name, 'o', 'O') AS bigO
FROM mypokemon;


SELECT name, UPPER(CONCAT(LEFT(type, 1),  RIGHT(type, 1))) AS type_code
FROM mypokemon;


SELECT *
FROM mypokemon
WHERE LENGTH(name) > 8;


SELECT ROUND(AVG(attack), 0) AS avg_of_attack
FROM mypokemon;


SELECT FLOOR(AVG(defense)) AS avg_of_defense
FROM mypokemon;


SELECT name, POWER(attack, 2) AS attack2
FROM mypokemon
WHERE LENGTH(name) < 8;


SELECT name, MOD(attack, 2) AS div2
FROM mypokemon;



SELECT name, ABS(attack-defense) AS diff
FROM mypokemon
WHERE attack <= 50;


SELECT CURRENT_DATE() AS now_date, CURRENT_TIME() AS now_time
FROM mypokemon;



SELECT DAYNAME(capture_date) AS day_eng, DAYOFWEEK(capture_date) AS day_num
FROM mypokemon;



SELECT YEAR(capture_date) AS year, MONTH(capture_date) AS month, DAY(capture_date) AS day
FROM mypokemon;


DROP DATABASE IF EXISTS pokemon;
CREATE DATABASE pokemon;
USE pokemon;
CREATE TABLE mypokemon (
	number  int,
       name	varchar(20),
       type	varchar(10),
       height	float,
       weight	float
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


SELECT type, AVG(height), AVG(weight)
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


SELECT type ,SUM(number)
FROM mypokemon
GROUP BY type;


SELECT type ,COUNT(1)
FROM mypokemon
WHERE height >= 0.5
GROUP BY type;



SELECT type ,MIN(height)
FROM mypokemon
GROUP BY type;

SELECT type ,MAX(weight)
FROM mypokemon
GROUP BY type;


SELECT type
FROM mypokemon
GROUP BY type
HAVING MIN(height) > 0.5 AND MAX(weight) < 30;



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
    WHEN number <200 THEN '<200'
    ELSE '<500'

END AS nunber_bin


FROM mypokemon;



SELECT name,
CASE
	WHEN number >= 150 AND attack >=50 THEN 'new_strong'
    WHEN number >= 150 AND attack < 50 THEN 'new_weak'
    WHEN number < 150 AND attack >=50 THEN 'old_strong'
    WHEN number < 150 AND attack < 50 THEN 'old_weak'

END AS age_attack
FROM mypokemon;


SELECT type,
	CASE
		WHEN COUNT(1) = 1 THEN 'solo'
        WHEN COUNT(1) < 3 THEN 'minor'
        ELSE 'major'
    END AS count_by_type

FROM mypokemon
GROUP BY type;


-- MISSION
-- 공격력과 방어력의 합이 120보다 크면 ‘very strong’, 90보다 크면 ‘strong’,
-- 모두 해당 되지 않으면 ‘not strong’를 반환하는 함수 ‘isStrong’을 만들고 사용해주세요.
-- 조건1: attack과 defense를 입력값으로 사용하세요.
-- 조건2: 결과값 데이터 타입은 VARCHAR(20)로 해주세요

SET GLOBAL log_bin_trust_function_creators = 1;

DELIMITER //


CREATE FUNCTION isStrong(attack INT, defense INT)
	RETURNS VARCHAR(20)
BEGIN
	DECLARE a INT;
	DECLARE b INT;
    DECLARE strong VARCHAR(20);
    SET a = attack;
    SET b = defense;
    SELECT CASE
		WHEN a+b > 120 THEN 'very_strong'
		WHEN a+b > 90 THEN 'strong'
		ELSE 'not_strong'
		END INTO strong;

	RETURN strong;
    
END

//
DELIMITER ; -- 꼭 한 칸을 띄어주세요!!


SELECT name, isStrong(attack, defense) AS isStrong
FROM mypokemon;
 
