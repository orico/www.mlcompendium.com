# Data contract

1. A data contract is **a formal agreement between a service and a client that abstractly describes the data to be exchanged**. That is, to communicate, the client and the service do not have to share the same types, only the same data contracts. - [Microsoft](https://learn.microsoft.com/en-us/dotnet/framework/wcf/feature-details/using-data-contracts)
2. The following is a set of articles around the topics of data product / contract etc. however the focus is about data contracts and expectations IMO, therefore I place it here. - by Chad sanderson
   1. [the existential threat of data quality](https://dataproducts.substack.com/p/the-existential-threat-of-data-quality)
   2. [the death of data modeling part 1](https://dataproducts.substack.com/p/the-death-of-data-modeling-pt-1)
   3. [data's collaboration problem](https://dataproducts.substack.com/p/datas-collaboration-problem)
   4. the rise of [data contracts](https://dataproducts.substack.com/p/the-rise-of-data-contracts)
   5. [production grade data products](https://dataproducts.substack.com/p/the-production-grade-data-pipeline)
   6. (finally!) [a guide to data contracts p1](https://dataproducts.substack.com/p/an-engineers-guide-to-data-contracts)

## Implementation

1. ****[**JSON Schema**](https://json-schema.org/) is a vocabulary that allows you to **annotate** and **validate** JSON documents, [example](https://json-schema.org/learn/miscellaneous-examples.html).
2. protobuf & gRPC
   1. [Protobuf on git](https://github.com/protocolbuffers/protobuf/tree/main/python), [google dev](https://developers.google.com/protocol-buffers)
   2. [what is gRPC?](https://grpc.io/docs/what-is-grpc/introduction/)
   3. [what are proto buffers, what do they solve and what are the benefits?](https://developers.google.com/protocol-buffers/docs/overview)
   4. gRPC - [a basic pythonic tutorial](https://grpc.io/docs/languages/python/basics/)
   5. Protobuf - [a pythonic tutorial](https://developers.google.com/protocol-buffers/docs/pythontutorial)
   6. (good) Intro to gRPC & Protobuf by [Trevor Kendrick](https://medium.com/@trevor.kendrick?source=post\_page-----c21054ef579c--------------------------------)
   7. (good) What are Protocol Buffers and why they are widely used? by [Dineshchandgr](https://medium.com/@dineshchandgr?source=post\_page-----cbcb04d378b6--------------------------------)
   8. [Understanding protobufs ](https://medium.com/danielpadua/understanding-protocol-buffers-protobuf-a466d8943df8)by [Daniel Padua Ferreira](https://medium.com/@danielpadua?source=post\_page-----a466d8943df8--------------------------------)\
      "Protocol Buffers (protobuf) is a method of serializing structured data which is particulary useful to communication between services or storing data.\
      It was designed by Google early 2001 (but only publicly released in 2008) to be smaller and faster than XML. Protobuf messages are serialized into a [binary wire](https://developers.google.com/protocol-buffers/docs/encoding) format which is very compact and boosts performance."
   9. protobuf what and why? by [Swaminathan Muthuveerappan](https://medium.com/@swamim?source=post\_page-----fcb324a64564--------------------------------)
   10. off topic - [how to choose between grpc, graphql, rest](https://ashish-bania.medium.com/the-exhaustive-guide-to-choosing-between-grpc-graphql-and-rest-b7e4fd6d547e)
3. managing proto files and other schema types such as avro or json schema, can be done in [kafka's schema registry](https://docs.confluent.io/platform/current/schema-registry/index.html).
