# SQL

A warehouse only answers if you can ask it in SQL, so this page is the command groups, keys, indexes, and how sharding differs from partitioning.
It stays one beat: asking a warehouse a question, starting with what kinds of commands exist, then the keys and indexes that make a question fast, then how a large table is split.

The same notes are in [Course](../analytics/data-analytics.md#course).

Every SQL statement belongs to one of five command groups, and knowing the group tells you whether you are describing, reading, changing, permitting, or committing data. The (very good) [DDL, DQL, DML, DCL and TCL](https://www.geeksforgeeks.org/sql-ddl-dql-dml-dcl-tcl-commands/), by GeeksforGeeks, the all-in-one learning portal, lists them with examples:

- DDL, Data Definition Language: create, drop, alter, truncate.
- DQL, Data Query Language: select.
- DML, Data Manipulation Language: update, insert, delete, lock.
- DCL, Data Control Language: grant, revoke.
- TCL, Transaction Control Language: commit, rollback, savepoint, set transaction.

With the groups named, the fastest way to see them in use is a crash course. [Introduction, index, keys, joins, aliases, and so on](https://www.youtube.com/watch?v=nWeW3sCmD2k) is Traversy Media's SQL Crash Course, beginner to intermediate, and the [newer](https://www.youtube.com/watch?v=9ylj9NR0Lcg) one is their MySQL Crash Course. The [SQL cheat sheet](https://gist.github.com/bradtraversy/c831baaad44343cc945e76c2e30927b3) is the MySQL Cheat Sheet gist to keep beside it.

The course touches keys and indexes; they deserve their own look because they decide how tables connect and how fast a question comes back. [Primary key](https://www.eukhost.com/blog/webhosting/whats-the-purpose-use-primary-foreign-keys/) is the purpose of primary and foreign keys. [Foreign key, a key constraint that is included in the primary key's allowed values](https://www.1keydata.com/sql/sql-foreign-key.html) shows how foreign key constraints link tables and enforce referential integrity, with CREATE TABLE and ALTER TABLE syntax for MySQL, Oracle, and SQL Server and composite key examples. [Index, that is, a book index for fast reading](https://www.tutorialspoint.com/sql/sql-indexes.htm) describes indexes as special lookup tables holding pointers to the data, so records are located faster.

Indexes help until the table itself is too large for one place. [Sharding vs partitioning](https://planetscale.com/learn/articles/sharding-vs-partitioning-whats-the-difference) is PlanetScale on the two common ways to improve the performance, manageability, and availability of larger databases, and how they differ.
