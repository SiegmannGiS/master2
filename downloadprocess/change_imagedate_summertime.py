import calendar
import datetime
import os

path = "C:\Master\images/vernagtferner14-16"

def summertime(year):

    stime = []
    a = (2,3)
    for i,month in enumerate((3, 10)):
        last_sunday = max(week[-1] for week in calendar.monthcalendar(year, month))
        stime.append(datetime.datetime(year=year, month=month, day=last_sunday, hour=a[i]))
    return stime

print summertime(2016)


for file in os.listdir(path):
    date = datetime.datetime.strptime("-".join(file.split("_")[:2]), "%Y-%m-%d-%H-%M")
    summer = summertime(date.year)
    if summer[0] < date < summer[1]:
        date = date + datetime.timedelta(hours=01)
        filename = datetime.datetime.strftime(date, "%Y-%m-%d_%H-%M")
        filename = filename+"_"+file.split("_")[-1]
        print file, filename



#if date1 < yourdate < date2: