# Leakage Removal Report

- Train file: `data/raw/imdb_train.parquet`
- Test file: `data/raw/imdb_test.parquet`
- Text column used: `text`

## Summary

- Unique texts in train: **24904**
- Unique texts in test: **24801**
- Exact overlap unique texts (train ∩ test): **123**
- Test rows removed (leakage rows): **123**
- Test rows before: **25000**
- Test rows after: **24877**

## Example overlapping texts (first 5)

1. Well the previews looked funny and I usually don't go to movies on opening night especially with my kids because ......well you never know. Here is a movie that doesn't appeal either to children or ad…
2. There is no way to avoid a comparison between The Cat in the Hat and The Grinch Who Stole Christmas, so let's get that part out of the way. First of all, let me start by saying that I think Grinch was…
3. We brought this film as a joke for a friend, and could of been our worst joke to play. The film is barely watchable, and the acting is dire. The worst child actor ever used and Hasslehoff giving a sub…
4. There's something frustrating about watching a movie like 'Murder By Numers' because somewhere inside that Hollywood formula is a good movie trying to pop out. However, by the time the credits roll, t…
5. The threesome of Bill Boyd, Robert Armstrong, and James Gleason play Coney Island carnys vying for the hand of Ginger Rogers, a working gal who sells salt water taffy. With the outbreak of World War I…