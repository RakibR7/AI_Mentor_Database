# Stack Overflow SQL Server Database - Mini 2010 Version

For more information and the latest release:
http://www.brentozar.com/go/querystack

Imported from the Stack Exchange Data Dump circa 2010
https://archive.org/details/stackexchange

Imported using the Stack Overflow Data Dump Importer:
https://github.com/BrentOzarULTD/soddi

This database is in Microsoft SQL Server 2008 format, which means you can
attach it to any SQL Server 2008 or newer instance.

To keep the size small but let you get started fast:

* All tables have a clustered index
* No nonclustered or full text indexes are included
* The log file is small, and you should grow it out if you plan to modify data
* It's distributed as an mdf/ldf so you don't need space to restore it
* It only includes StackOverflow.com data, not data for other Stack sites

As with the original data dump, this is provided under cc-by-sa 3.0 license:
http://creativecommons.org/licenses/by-sa/3.0/

You are free to share this database and adapt it for any purpose, even
commercially, but you must attribute it to the original authors:
https://archive.org/details/stackexchange