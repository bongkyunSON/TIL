-- 1. 2020년 7월의 총 Revenue를 구해주세요 


select sum(price)
from sql.tbl_purchase
where purchased_at >= '2020-07-01' and 
	purchased_at < '2020-08-01';


-- 2. 2020년 7월의 MAU를 구해주세요
select count(distinct customer_id)
from sql.tbl_visit
where visited_at >= '2020-07-01' and 
	visited_at < '2020-08-01';
    
    


-- 3. 7월에 우리 Acitive 유저의 구매율은 어떻게 되나요? 
select count(distinct customer_id)
from sql.tbl_purchase
where purchased_at >= '2020-07-01' and
	purchased_at < '2020-08-01';
    
    
select count(distinct customer_id)
from sql.tbl_visit
where visited_at >= '2020-07-01' and 
	visited_at < '2020-08-01';
    

select round(11174/16414 *100, 2);




-- 4. 7월에 구매유저의 월 평균 구매금액은 어떻게 되나요?
select avg(revenue)
from (select customer_id, sum(price) as revenue
	from sql.tbl_purchase
	where purchased_at >= '2020-07-01' and
		purchased_at < '2020-08-01'
        group by 1) as a;
        
        
        
        
-- 5. 7월에 가장 많이 구매한 고객 top3와 top10~15 고객을 뽑아주세요

select customer_id, sum(price)
from sql.tbl_purchase
where purchased_at >= '2020-07-01' and
	purchased_at < '2020-08-01'
group by 1
order by 2 desc
limit 5 offset 10;


-- 6. 2020년 7월의 평균 DAU를 구해주세요 Active User 수가 증가하는 추세인가요?

select avg(users)
from(select date_format(visited_at - interval 9 hour, '%Y-%m-%d') as date_at, 
	count(distinct customer_id) as users
from sql.tbl_visit
where visited_at >= '2020-07-01' and
	visited_at < '2020-08-01'
group by 1
order by 1) as a;


-- 7.  2020년 7월의 평균 WAU를 구해주세요


select avg(users)
from(
select date_format(visited_at - interval 9 hour, '%Y-%m-%U') as date_at, 
	count(distinct customer_id) as users
from sql.tbl_visit
where visited_at >= '2020-07-05' and
	visited_at < '2020-07-26'
group by 1
order by 1) as a;



--  8. 2020년 7월 daily revenue는 증가 추세인가요? 평균 daily revenue도 구해주세요

select avg(revenue)
from(
select date_format(purchased_at -interval 9 hour, '%Y-%m-%d'),
		sum(price) as revenue
from sql.tbl_purchase
where purchased_at >= '2020-07-01' and
		purchased_at < '2020-08-01'
group by 1
order by 1) as a;


-- 9. 7월 평균 weekly revenue를 구해주세요 
select avg(revenue)
from(
select date_format(purchased_at -interval 9 hour, '%Y-%m-%U'),
		sum(price) as revenue
from sql.tbl_purchase
where purchased_at >= '2020-07-05' and
		purchased_at < '2020-07-26'
group by 1
order by 1) as a;


-- 10. 7월 요일별 revenue를 구해주세요
select date_format(date_at, '%w') as day_order,
		date_format(date_at, '%W') as day_name,
        avg(revenue)
from(
select date_format(purchased_at -interval 9 hour, '%Y-%m-%d') as date_at,
		sum(price) as revenue
from sql.tbl_purchase
where purchased_at >= '2020-07-05' and
		purchased_at < '2020-08-01'
group by 1) as a
group by 1, 2   
order by 1; 



-- 11. 7월에 시간대별 revenue를 구해주세요
select hour_at, avg(revenue)
from(
select date_format(purchased_at -interval 9 hour, '%Y-%m-%d') as date_at,
		date_format(purchased_at -interval 9 hour, '%H') as hour_at,
		sum(price) as revenue
from sql.tbl_purchase
where purchased_at >= '2020-07-05' and
		purchased_at < '2020-08-01'
group by 1,2) as a
group by 1
order by 2 desc;


-- 12. 7월의 요일 및 시간대별 revenue를 구해주세요 
select dayname_at, hour_at ,avg(revenue)
from(
select date_format(purchased_at -interval 9 hour, '%Y-%m-%d') as date_at,
		date_format(purchased_at -interval 9 hour, '%W') as dayname_at,
		date_format(purchased_at -interval 9 hour, '%H') as hour_at,
		sum(price) as revenue
from sql.tbl_purchase
where purchased_at >= '2020-07-01' and
		purchased_at < '2020-08-01'
group by 1,2,3) as a
group by 1,2
order by 3 desc


        
        
        



    