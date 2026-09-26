# Data Contract

This page defines data contracts and points at articles and implementation options.

1. A data contract is **a formal agreement between a service and a client that abstractly describes the data to be exchanged**. That is, to communicate, the client and the service do not have to share the same types, only the same data contracts. — [Microsoft](https://learn.microsoft.com/en-us/dotnet/framework/wcf/feature-details/using-data-contracts)
2. The following is a set of articles around the topics of data product / contract etc. however the focus is about data contracts and expectations IMO, therefore I place it here. — by Chad Sanderson
   - [the existential threat of data quality](https://dataproducts.substack.com/p/the-existential-threat-of-data-quality)
   - [the death of data modeling part 1](https://dataproducts.substack.com/p/the-death-of-data-modeling-pt-1)
   - [data's collaboration problem](https://dataproducts.substack.com/p/datas-collaboration-problem)
   - the rise of [data contracts](https://dataproducts.substack.com/p/the-rise-of-data-contracts)
   - [production grade data products](https://dataproducts.substack.com/p/the-production-grade-data-pipeline)
   - (finally!) [a guide to data contracts p1](https://dataproducts.substack.com/p/an-engineers-guide-to-data-contracts)
   - (good) [contracts, robustness in datamesh (but not only)](https://medium.com/data-science/data-contracts-ensure-robustness-in-your-data-mesh-architecture-69a3c38f07db), Piethein Strengholt — also has about data-sharing agreements.
   - (good) [from zero to hero](https://medium.com/data-science/data-contracts-from-zero-to-hero-343717ac4d5e), mehdio
   - [data contract experimentation](https://www.newsletter.swirlai.com/p/sai-04-data-contracts-experimentation) — swirlai

## Implementation

This section covers JSON Schema, protobuf, gRPC, and schema registries for managing contracts.

- [**JSON Schema**](https://json-schema.org/) is a vocabulary that allows you to **annotate** and **validate** JSON documents, [example](https://json-schema.org/learn/miscellaneous-examples.html).
- protobuf & gRPC
   - [Protobuf on git](https://github.com/protocolbuffers/protobuf/tree/main/python), [google dev](https://developers.google.com/protocol-buffers)
   - [what is gRPC?](https://grpc.io/docs/what-is-grpc/introduction/)
   - [what are proto buffers, what do they solve and what are the benefits?](https://developers.google.com/protocol-buffers/docs/overview)
   - gRPC — [a basic pythonic tutorial](https://grpc.io/docs/languages/python/basics/)
   - Protobuf — [a pythonic tutorial](https://developers.google.com/protocol-buffers/docs/pythontutorial)
   - (good) Intro to gRPC & Protobuf by [Trevor Kendrick](https://medium.com/@trevor.kendrick?source=post_page-----c21054ef579c--------------------------------)
   - (good) What are Protocol Buffers and why they are widely used? by [Dineshchandgr](https://medium.com/@dineshchandgr?source=post_page-----cbcb04d378b6--------------------------------)
   - [Understanding protobufs](https://medium.com/danielpadua/understanding-protocol-buffers-protobuf-a466d8943df8) by [Daniel Padua Ferreira](https://medium.com/@danielpadua?source=post_page-----a466d8943df8--------------------------------)

> Protocol Buffers (protobuf) is a method of serializing structured data which is particulary useful to communication between services or storing data.
> It was designed by Google early 2001 (but only publicly released in 2008) to be smaller and faster than XML. Protobuf messages are serialized into a [binary wire](https://developers.google.com/protocol-buffers/docs/encoding) format which is very compact and boosts performance.

   - protobuf what and why? by [Swaminathan Muthuveerappan](https://medium.com/@swamim?source=post_page-----fcb324a64564--------------------------------)
   - off topic — [how to choose between grpc, graphql, rest](https://ashish-bania.medium.com/the-exhaustive-guide-to-choosing-between-grpc-graphql-and-rest-b7e4fd6d547e)
- Managing proto files and other schema types such as avro or json schema can be done in [kafka's schema registry](https://docs.confluent.io/platform/current/schema-registry/index.html).

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- (good) contracts, robustness in datamesh (but not only). This address no longer opens: https://towardsdatascience.com/data-contracts-ensure-robustness-in-your-data-mesh-architecture-69a3c38f07db
- (good) from zero to hero. This address no longer opens: https://towardsdatascience.com/data-contracts-from-zero-to-hero-343717ac4d5e
