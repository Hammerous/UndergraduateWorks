import requests
import pymysql

url='https://network.igs.org/api/public/stations/'

cookies = {
    #'csrftoken': '47Aj3o9aNJUfbJ7peaVhnWVjDHBA7Eqq',
    #'_ga_Z5RH7R682C': 'GS1.1.1682676353.8.0.1682676353.60.0.0',
    #'_ga': 'GA1.2.1472907361.1678196092',
    #'_gid': 'GA1.2.77720386.1682676354',
    'csrftoken':'HZ1gGcZIrLJG6gsvfsJaPU2RAXGeWpx',  ##根据浏览器上的token更新
    '_gid':'GA1.2.668211834.1695049609',
    '_ga_Z5RH7R682C':'GS1.1.1695049608.1.1.1695050465.60.0.0',
    '_ga':'GA1.2.2105865382.1695049609; _gat_UA-215834315-1=1',
}
headers = {
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    # 'Cookie': 'csrftoken=47Aj3o9aNJUfbJ7peaVhnWVjDHBA7Eqq; _ga_Z5RH7R682C=GS1.1.1682676353.8.0.1682676353.60.0.0; _ga=GA1.2.1472907361.1678196092; _gid=GA1.2.77720386.1682676354',
    'Pragma': 'no-cache',
    'Referer': 'https://network.igs.org/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.31',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua': '"Chromium";v="112", "Microsoft Edge";v="112", "Not:A-Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}
#//t=now()
#def
data = {
    'draw': '2',
    'length': '1000',
    'current': 'true',
    'ordering': 'name',
    '_': '1695049657289',       ##根据浏览器上的token更新
    ### 注：浏览器可不带“_”后缀访问，后续爬虫可尝试去除该参数
}
print()
response = requests.get(url, params=data, cookies=cookies, headers=headers).json()

print((response.get('data'))[1].get('name'))

#连接数据库
con=pymysql.connect(host='localhost',user='root',password='Pathfinder',database='db3',charset='utf8')

# 创建游标
cur=con.cursor()

# 创建数据库
#cur.execute("create database if not exists db3 default charset utf8 collate utf8_general_ci")

#con.commit()

cur.execute("use db3")

#创建测站信息数据表
sql="""create table if not exists `测站信息`(
    `Site-Name` char(9) not null primary key ,
    `Country/Region` varchar(60),
    `Receiver` varchar(20),
    `Antenna` varchar(30),
    `Radome` char(4),
    `Satellite-System` varchar(40),
    `Latitude` decimal(35,30),
    `Longitude` decimal(35,30),
    `Height(m)` decimal(15,10),
    `X(m)` decimal(12,3),
    `Y(m)` decimal(12,3),
    `Z(m)` decimal(12,3),
    `Calibration` varchar(10) null,
    `Networks`  varchar(100),
    `Data-Center` varchar(100),
    `IERS-DOMES` varchar(9),
    `Clock` varchar(60) null,
    `Agencies` varchar(50),
    `Last-Data` int
)default charset=utf8"""

cur.execute(sql)

con.commit()

usersvalues1 = []
max_num = int(response.get('recordsFiltered'))
for num in range(max_num):
    a=''
    #print(len((response.get('data'))[num].get('satellite_system')))
    for num1 in range(len((response.get('data'))[num].get('satellite_system'))):
        if num1==0:
            a=a+(response.get('data'))[num].get('satellite_system')[num1]
        else:
            a=a+'+'+(response.get('data'))[num].get('satellite_system')[num1]
    b=''
    for num1 in range(len((response.get('data'))[num].get('networks'))):
        if num1==0:
            b=b+(response.get('data'))[num].get('networks')[num1].get('name')
        else:
            b=b+','+(response.get('data'))[num].get('networks')[num1].get('name')
    usersvalues1.append((((response.get('data'))[num].get('name'),
                          (response.get('data'))[num].get('country'),                        
                          (response.get('data'))[num].get('receiver_type'),
                          (response.get('data'))[num].get('antenna_type'),
                          (response.get('data'))[num].get('radome_type'),
                          a,
                          ((response.get('data'))[num].get('llh'))[0],
                          ((response.get('data'))[num].get('llh'))[1],
                          ((response.get('data'))[num].get('llh'))[2],
                          ((response.get('data'))[num].get('xyz'))[0],
                          ((response.get('data'))[num].get('xyz'))[1],
                          ((response.get('data'))[num].get('xyz'))[2],
                          (response.get('data'))[num].get('antcal'),
                          b,
                          ((response.get('data'))[num].get('agencies'))[0].get('shortname'),
                          (response.get('data'))[num].get('domes_number'),
                          (response.get('data'))[num].get('frequency_standard'),
                          ((response.get('data'))[num].get('agencies'))[0].get('shortname'),
                          (response.get('data'))[num].get('last_data')
                          )))

cur.executemany('INSERT INTO `测站信息` values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)', usersvalues1)
con.commit()

# 获取所有记录 fetchall--获取所有记录 fetchmany--获取多条记录，需传参  fetchone--获取一条记录
#all=cur.fetchall()

#cur.execute("show tables")


# 关闭游标
cur.close()

# 关闭数据库连接，目的为了释放内存
cur.close()
