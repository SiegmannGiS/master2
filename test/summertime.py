import sys
import calendar
from datetime import datetime




def day(year):
    laststart_sunday = max(week[-1] for week in calendar.monthcalendar(year, 3))
    lastend_sunday = max(week[-1] for week in calendar.monthcalendar(year, 10))
    return datetime(year=year,month=3,day=laststart_sunday, hour=2),datetime(year=year,month=10,day=laststart_sunday, hour=3)


datetime(year=2016,month=1,day=1)

if day(2016)[0] < datetime.now() < day(2016)[1]:
    print day(2015)[0]
    print True