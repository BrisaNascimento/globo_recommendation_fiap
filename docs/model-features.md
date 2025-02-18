# Features

The created ML model workes with the following features:

1. **Input Features:** As input feature our API receives a user ID.
2. **Internal calculation:** Once the user ID is provided to the API, the service will retrieve the last 5 news consumed by the given user. If the user has a history then it will use this history to calculate the cossine similarity with the latest news available and if the user is a new user, then the API will capture a rank of the most viewed news.
3. **Output Features:** As output feature we are going to provide a json with the following data:

```json
{"userId": "1",
 "preds":[
	{
		"page": "279ccbff-f203-4c6d-aa48-83d97f085302",
		"cossine_similarity": 0
	},
	{
		"page": "aeab0e46-f1e4-41e9-821b-571255c41f69",
		"cossine_similarity": 0
	},
	{
		"page": "d730c4a6-e8f6-4fde-b73a-afbe148479cd",
		"cossine_similarity": 0
	},
	{
		"page": "561850c2-9ade-4985-bf36-402abf02153d",
		"cossine_similarity": 0
	},
	{
		"page": "3c787cdc-99f7-4001-94ac-4edb2e971b94",
		"cossine_similarity": 0
	},
	{
		"page": "2b4fefsf-ab7b-45e7-8a8a-353e0aa41249",
		"cossine_similarity": 0
	}]
}
```

Note: you may observe that in some cases the API will return as cossine similarity the value 0, this happens when the user has no history, in this case we are going provide the rank with the 15 most viewed content and as such there is no similarity calculated yet.