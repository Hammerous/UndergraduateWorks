import pymysql

#连接数据库
con=pymysql.connect(host='localhost',user='root',password='Pathfinder',database='test1',charset='utf8')

# 创建游标
cur=con.cursor()

# 创建数据库
cur.execute("create database if not exists db3 default charset utf8 collate utf8_general_ci")

con.commit()

cur.execute("use db3")

#创建测站信息数据表
sql="""create table if not exists `测站信息`(
    `Site-Name` char(9) not null primary key ,
    `Country/Region` varchar(10),
    `Receiver` varchar(20),
    `Radome` char(4),
    `Satellite-System` varchar(40),
    `Latitude` decimal(35,30),
    `Longitude` decimal(35,30),
    `Height(m)` decimal(15,10),
    `X(m)` decimal(12,10),
    `Y(m)` decimal(12,10),
    `Z(m)` decimal(12,10),
    `Calibration` varchar(10) null,
    `network`  varchar(40),
    `Data-Center` varchar(100),
    `IERS-DOMES` varchar(9),
    `Clock` varchar(9) null,
    `Agencies` varchar(50),
    `Last-Data` int
)default charset=utf8"""

cur.execute(sql)

con.commit()

# 获取所有记录 fetchall--获取所有记录 fetchmany--获取多条记录，需传参  fetchone--获取一条记录
all=cur.fetchall()

cur.execute("show tables")

# 关闭游标
cur.close()

# 关闭数据库连接，目的为了释放内存
cur.close()
