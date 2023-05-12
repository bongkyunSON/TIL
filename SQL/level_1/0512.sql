DROP DATABASE IF EXISTS pokemon;
CREATE DATABASE pokemon;
USE pokemon;
CREATE TABLE mypokemon (
	   number  INT,
       name	VARCHAR(20),
       type	VARCHAR(10)
);
INSERT INTO mypokemon (number, name, type)
VALUES (10, 'caterpie', 'bug'),
	   (25, 'pikachu', 'electric'),
       (26, 'raichu', 'electric'),
       (133, 'eevee', 'normal'),
       (152, 'chikoirita', 'grass');
CREATE TABLE ability (
	   number INT,
       height FLOAT,
       weight FLOAT,
       attack INT,
       defense INT,
       speed int
);
INSERT INTO ability (number, height, weight, attack, defense, speed)
VALUES (10, 0.3, 2.9, 30, 35, 45),
	   (25, 0.4, 6, 55, 40, 90),
       (125, 1.1, 30, 83, 57, 105),
	   (133, 0.3, 6.5, 55, 50, 55),
       (137, 0.8, 36.5, 60, 70, 40),
	   (152, 0.9, 6.4, 49, 65, 45),
       (153, 1.2, 15.8, 62, 80, 60),
       (172, 0.3, 2, 40, 15, 60),
       (470, 1, 25.5, 110, 130, 95);
       
       
       
       
SELECT *
FROM mypokemon
LEFT JOIN ability
ON mypokemon.number = ability.number;
    


SELECT mypokemon.number, name
FROM mypokemon
RIGHT JOIN ability
ON mypokemon.number = ability.number;




SELECT type, AVG(height)
FROM mypokemon
LEFT JOIN ability
ON mypokemon.number = ability.number
GROUP BY type;

    
SELECT type, AVG(weight)
FROM mypokemon
LEFT JOIN ability
ON mypokemon.number = ability.number
GROUP BY type; 


SELECT type, AVG(weight), AVG(weight)
FROM mypokemon
LEFT JOIN ability
ON mypokemon.number = ability.number
GROUP BY type; 



SELECT mypokemon.number, name, attack, defense
FROM mypokemon
LEFT JOIN ability
ON mypokemon.number = ability.number
WHERE mypokemon.number > 100;


    
SELECT mypokemon.name
FROM mypokemon
LEFT JOIN ability
ON mypokemon.number = ability.number
ORDER BY attack + defense DESC;
    



SELECT mypokemon.name
FROM mypokemon
LEFT JOIN ability
ON mypokemon.number = ability.number
ORDER BY speed DESC
LIMIT 1;
    





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
CREATE TABLE friendpokemon (
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
	   (133, 'eevee', 'normal', 55, 50),
	   (152, 'chikoirita', 'grass', 49, 65);

INSERT INTO friendpokemon (number, name, type, attack, defense)
VALUES (26, 'raichu', 'electric', 80, 60),
	   (125, 'electabuzz', 'electric', 83, 57),
       (137, 'porygon', 'normal', 60, 70),
       (153, 'bayleef', 'grass', 62, 80),
       (172, 'pichu', 'electric', 40, 15),
       (470, 'leafeon', 'grass', 110, 130);
       
       



SELECT a.type
FROM mypokemon AS a
UNION
SELECT b.type
FROM friendpokemon AS b;



SELECT a.number, a.name
FROM mypokemon AS a
WHERE type = 'grass'
UNION ALL
SELECT b.number, b.name
FROM friendpokemon AS b
WHERE type = 'grass';


SELECT a.name
FROM mypokemon AS a
INNER JOIN friendpokemon AS b
ON a.name = b.name;



SELECT a.name
FROM mypokemon AS a
LEFT JOIN friendpokemon AS b
ON a.name = b.name
WHERE b.name IS NULL;





DROP DATABASE IF EXISTS pokemon;
CREATE DATABASE pokemon;
USE pokemon;
CREATE TABLE mypokemon (
	   number  INT,
       name	VARCHAR(20)
);
INSERT INTO mypokemon (number, name)
VALUES (10, 'caterpie'),
	   (25, 'pikachu'),
       (26, 'raichu'),
       (133, 'eevee'),
       (152, 'chikoirita');
CREATE TABLE ability (
	   number INT,
	   type	VARCHAR(10),
       height FLOAT,
       weight FLOAT,
       attack INT,
       defense INT,
       speed int
);
INSERT INTO ability (number, type, height, weight, attack, defense, speed)
VALUES (10, 'bug', 0.3, 2.9, 30, 35, 45),
	   (25, 'electric', 0.4, 6, 55, 40, 90),
       (26, 'electric', 0.8, 30, 90, 55, 110),
	   (133, 'normal', 0.3, 6.5, 55, 50, 55),
	   (152, 'grass', 0.9, 6.4, 49, 65, 45);


SELECT number
FROM mypokemon
WHERE (SELECT weight FROM ability WHERE MAX(weight));




SELECT a.number
FROM mypokemon AS a
INNER JOIN ability AS b
ON a.number = b.number
ORDER BY weight DESC
LIMIT 1;


SELECT number
FROM ability
WHERE weight = (SELECT MAX(weight) FROM ability);


SELECT number
FROM ability
WHERE speed < ANY(SELECT attack FROM ability WHERE type = 'electric');



SELECT a.name
FROM mypokemon AS a
LEFT JOIN ability AS b
ON a.number = b.number
WHERE b.attack > b.defense;


SELECT name
FROM mypokemon
WHERE EXISTS(SELECT * FROM ability WHERE attack > defense);





--


/*
MISSION (1)
이브이의 번호 133을 활용해서, 이브이의 영문 이름, 키, 몸무게를 가져와 주세요.
이 때, 키는 height, 몸무게는 weight이라는 별명으로 가져와 주세요. 
*/

SELECT name, 
	(SELECT height FROM ability WHERE number = 133) AS height,
    (SELECT weight FROM ability WHERE number = 133) AS weight
FROM mypokemon
WHERE number = 133;




/*
MISSION (2)
속도가 2번째로 빠른 포켓몬의 번호와 속도를 가져와 주세요.
*/
SELECT number, speed
FROM (SELECT number, speed, RANK() OVER(ORDER BY speed DESC) AS speed_rank FROM ability) AS a
WHERE speed_rank =2;




/*
MISSION (3)
방어력이 모든 전기 포켓몬의 방어력보다 큰 포켓몬의 이름을 가져와 주세요.
*/
SELECT name
FROM mypokemon 
WHERE number = (SELECT number FROM ability WHERE defense > (SELECT MAX(defense) FROM ability WHERE type = 'electric'));



