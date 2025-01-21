SELECT 
    q.Id AS QuestionId,
    q.Title AS QuestionTitle,
    q.Body AS QuestionBody,
    q.Tags AS Tags,
    a.Id AS AnswerId,
    a.Body AS AnswerBody,
    a.Score AS AnswerScore
FROM Posts q
JOIN Posts a ON q.Id = a.ParentId
WHERE q.PostTypeId = 1 -- Questions
  AND a.PostTypeId = 2 -- Answers
  AND q.Score > 0      -- Filter for high-quality questions (optional)
  AND a.Score > 0      -- Filter for high-quality answers (optional)
ORDER BY q.Score DESC, a.Score DESC; -- Sort by question and answer quality
