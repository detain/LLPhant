<?php

namespace LLPhant\Embeddings\VectorStores\Pinecone;

use LLPhant\Embeddings\Document;
use LLPhant\Embeddings\DocumentStore\DocumentStore;
use LLPhant\Embeddings\VectorStores\VectorStoreBase;
use LLPhant\Exception\SecurityException;

class PineconeVectorStore extends VectorStoreBase implements DocumentStore
{
    public const DEFAULT_NAMESPACE = '';

    public function __construct(
        public PineconeClient $client,
        public string $namespace = self::DEFAULT_NAMESPACE
    ) {
    }

    public function addDocument(Document $document): void
    {
        $this->addDocuments([$document]);
    }

    /**
     * @param  Document[]  $documents
     */
    public function addDocuments(array $documents): void
    {
        if ($documents === []) {
            return;
        }

        $vectors = array_map(
            fn (Document $document): array => [
                'id' => $document->hash,
                'values' => $document->embedding,
                'metadata' => [
                    'content' => $document->content,
                    'formattedContent' => $document->formattedContent ?? $document->content,
                    'sourceType' => $document->sourceType,
                    'sourceName' => $document->sourceName,
                    'hash' => $document->hash,
                    'chunkNumber' => $document->chunkNumber,
                ],
            ],
            $documents
        );

        $success = $this->client->upsert($vectors, $this->namespace);

        if (! $success) {
            throw new \Exception('Failed to upsert vectors to Pinecone.');
        }
    }

    /**
     * {@inheritDoc}
     *
     * @param  array{filter?: array<string, mixed>}  $additionalArguments
     */
    public function similaritySearch(array $embedding, int $k = 4, array $additionalArguments = []): array
    {
        $filter = $additionalArguments['filter'] ?? [];

        $matches = $this->client->query(
            vector: $embedding,
            topK: $k,
            filter: $filter,
            namespace: $this->namespace
        );

        return $this->mapMatchesToDocuments($matches);
    }

    /**
     * {@inheritDoc}
     */
    public function fetchDocumentsByChunkRange(string $sourceType, string $sourceName, int $leftIndex, int $rightIndex): iterable
    {
        if (trim($sourceType) === '' || trim($sourceName) === '') {
            throw new SecurityException('Invalid source type or name');
        }

        $stats = $this->client->describeIndexStats();
        $dimension = $stats['dimension'] ?? 1536;
        $zeroVector = array_fill(0, $dimension, 0.0);

        $filter = [
            'sourceType' => ['$eq' => $sourceType],
            'sourceName' => ['$eq' => $sourceName],
            'chunkNumber' => [
                '$gte' => $leftIndex,
                '$lte' => $rightIndex,
            ],
        ];
        $limit = $rightIndex - $leftIndex + 1;
        $matches = $this->client->query(
            vector: $zeroVector,
            topK: $limit,
            filter: $filter,
            namespace: $this->namespace
        );
        $documents = $this->mapMatchesToDocuments($matches);

        usort($documents, fn (Document $a, Document $b) => $a->chunkNumber <=> $b->chunkNumber);

        return $documents;
    }

    /**
     * @param  array<mixed>  $matches
     * @return Document[]
     */
    protected function mapMatchesToDocuments(array $matches): array
    {
        $documents = [];
        foreach ($matches as $match) {
            $metadata = $match['metadata'] ?? [];

            $doc = new Document();
            $doc->content = $metadata['content'] ?? '';
            $doc->formattedContent = $metadata['formattedContent'] ?? '';
            $doc->sourceType = $metadata['sourceType'] ?? '';
            $doc->sourceName = $metadata['sourceName'] ?? '';
            $doc->hash = $match['id'];
            $doc->chunkNumber = isset($metadata['chunkNumber']) ? (int) $metadata['chunkNumber'] : 0;

            $documents[] = $doc;
        }

        return $documents;
    }
}
